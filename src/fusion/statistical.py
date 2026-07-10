"""Layer-1 statistical fusion: per-channel classifiers + score-level late fusion.

Two responsibilities:

1. Build a per-channel `StandardScaler + LogisticRegression` pipeline that turns a
   subject's feature vector into a PD probability. LogReg is chosen for direct
   calibrated probabilities (feeds fusion + the Claude report) and to keep the
   model class aligned with the smile classifier.

2. Combine per-channel probabilities into a single fused score via a simple
   weighted average (weights from `configs/model.yaml`, renormalized over the
   channels actually present). Averaging is deliberately chosen over a learned
   meta-combiner because the LOSO evaluation only produces one out-of-fold
   probability per subject — too little data for a stable meta-learner at N≈95.

`eval/ablation.py` drives the LOSO loop; this module holds only pure, testable
helpers that the ablation script and Layer-2 demo both import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

PHONATION_FEATURES = [
    "jitter_local", "jitter_rap", "jitter_ppq5",
    "shimmer_local", "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
    "hnr_mean_db",
    "f0_mean_hz", "f0_std_hz", "f0_range_hz",
    "voicing_fraction",
]

DDK_FEATURES = [
    "n_peaks", "duration_s", "ddk_rate_hz",
    "isi_mean_s", "isi_cv",
    "amp_mean", "amp_cv", "amp_decrement",
]


@dataclass
class ChannelSpec:
    """Everything needed to train one channel end-to-end."""
    name: str
    features_csv: Path
    feature_cols: list[str]


def default_channel_specs(paths: dict) -> dict[str, ChannelSpec]:
    return {
        "phonation": ChannelSpec(
            name="phonation",
            features_csv=REPO_ROOT / paths["phonation_features_dir"] / "per_subject.csv",
            feature_cols=PHONATION_FEATURES,
        ),
        "ddk": ChannelSpec(
            name="ddk",
            features_csv=REPO_ROOT / paths["ddk_features_dir"] / "per_subject.csv",
            feature_cols=DDK_FEATURES,
        ),
    }


def load_paths(paths_yaml: Path = REPO_ROOT / "configs" / "paths.yaml") -> dict:
    with open(paths_yaml) as f:
        return yaml.safe_load(f)


def load_model_config(model_yaml: Path = REPO_ROOT / "configs" / "model.yaml") -> dict:
    with open(model_yaml) as f:
        return yaml.safe_load(f)


def load_analysis_cohort(paths: dict) -> pd.DataFrame:
    """Subject-level cohort table filtered to `in_analysis_cohort == True`.

    Provides subject_id → group (HC/PD) and other clinical labels.
    """
    cohort_path = REPO_ROOT / paths["cohort_csv"]
    df = pd.read_csv(cohort_path)
    return df[df["in_analysis_cohort"]].reset_index(drop=True)


def load_channel_matrix(
    spec: ChannelSpec, cohort: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join per-subject features onto the analysis cohort in cohort order.

    Returns (X, y, subject_ids) where y is 1 for PD, 0 for HC. Any subject in
    the cohort missing from the feature CSV raises — the cohort filter should
    have already guaranteed presence.
    """
    features = pd.read_csv(spec.features_csv)
    missing = set(cohort["subject_id"]) - set(features["subject_id"])
    if missing:
        raise ValueError(
            f"[{spec.name}] {len(missing)} cohort subjects missing from features: "
            f"{sorted(missing)[:5]}..."
        )
    merged = cohort[["subject_id", "group"]].merge(
        features[["subject_id", *spec.feature_cols]], on="subject_id", how="left"
    )
    X = merged[spec.feature_cols].to_numpy(dtype=float)
    y = (merged["group"] == "PD").astype(int).to_numpy()
    return X, y, merged["subject_id"].tolist()


def load_phonation_per_file(
    paths: dict, cohort: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load per-file phonation features restricted to the analysis cohort.

    Returns (X, y, subject_ids, tasks) where each row is one vowel file.
    `subject_ids` doubles as the LOSO group label — every file from the same
    subject shares the same group and must land together in either train or test.
    """
    csv_path = REPO_ROOT / paths["phonation_features_dir"] / "per_file.csv"
    features = pd.read_csv(csv_path)
    cohort_ids = set(cohort["subject_id"])
    features = features[features["subject_id"].isin(cohort_ids)].reset_index(drop=True)
    X = features[PHONATION_FEATURES].to_numpy(dtype=float)
    y = (features["group"] == "PD").astype(int).to_numpy()
    return X, y, features["subject_id"].tolist(), features["task"].tolist()


def make_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    """StandardScaler → LogReg(l2, C).

    Wrapping in a Pipeline means the scaler is re-fit inside every LOSO fold
    automatically — no scaler leakage from held-out subjects. LogReg gets a
    calibrated probability for free (`predict_proba`).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, solver="lbfgs", max_iter=2000, random_state=seed,
        )),
    ])


def make_pipeline_l1(C: float = 1.0, seed: int = 42) -> Pipeline:
    """StandardScaler → LogReg(l1, C) — feature-selection variant for viz only.

    L1 zeros out redundant features. Phonation has 4 shimmer variants and 3
    jitter variants that are near-collinear; L2 splits the coefficient across
    them with unstable signs (multicollinearity). L1 forces the coefficient
    onto a sparse representative subset, giving a readable coefficients bar
    plot. NOT used for deployment — the L2 pipeline is what feeds Layer 2.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", C=C, solver="liblinear", max_iter=2000, random_state=seed,
        )),
    ])


def fuse_probs(
    probs_by_channel: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Score-level late fusion.

    - If `weights` is None → unweighted average across channels present.
    - Otherwise → weighted average, weights renormalized over channels present
      (an N/A channel drops out cleanly at demo time).
    """
    if not probs_by_channel:
        raise ValueError("fuse_probs called with no channels")
    channels = list(probs_by_channel.keys())
    stacked = np.stack([probs_by_channel[c] for c in channels], axis=0)
    if weights is None:
        return stacked.mean(axis=0)
    w = np.array([weights.get(c, 0.0) for c in channels], dtype=float)
    if w.sum() <= 0:
        raise ValueError(f"Fusion weights sum to zero for channels {channels}")
    w = w / w.sum()
    return (stacked * w[:, None]).sum(axis=0)


def fit_final_channel_model(
    spec: ChannelSpec, cohort: pd.DataFrame, C: float = 1.0, seed: int = 42
) -> tuple[Pipeline, dict]:
    """Fit on the full analysis cohort and return the pipeline + metadata.

    Used to persist the deployment classifier for Layer 2 (demo time). The
    LOSO evaluation in `eval/ablation.py` fits fold-specific pipelines
    separately and does not call this.
    """
    X, y, subject_ids = load_channel_matrix(spec, cohort)
    pipe = make_pipeline(C=C, seed=seed)
    pipe.fit(X, y)
    meta = {
        "channel": spec.name,
        "n_subjects": int(len(y)),
        "n_pd": int(y.sum()),
        "n_hc": int((y == 0).sum()),
        "feature_cols": spec.feature_cols,
        "subject_ids": subject_ids,
        "C": C,
        "seed": seed,
    }
    return pipe, meta


def save_channel_model(
    pipe: Pipeline, meta: dict, models_dir: Path
) -> tuple[Path, Path]:
    """Persist a channel's deployment classifier + metadata for Layer 2."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{meta['channel']}.joblib"
    meta_path = models_dir / f"{meta['channel']}_meta.json"
    joblib.dump(pipe, model_path)
    import json
    meta_path.write_text(json.dumps(meta, indent=2))
    return model_path, meta_path
