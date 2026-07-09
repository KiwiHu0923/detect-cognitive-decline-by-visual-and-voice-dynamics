"""Train the facial-channel PD classifier.

Uses the ROC-HCI UFNet released feature dataset (1361 subjects, 42 session-level
features: 7 OpenFace AUs + 7 MediaPipe geometric signals, each aggregated as
{mean, var, entropy}) and their predefined participant splits (dev/test). The
rest of the participants form the training set. Matches UFNet's training recipe:
StandardScaler, SMOTE oversampling, and a linear model on the 42 features
(sklearn LogisticRegression here in place of their pytorch ShallowANN — same
model class mathematically, no torch dependency at inference).

External validation on the released YoutubePD features CSV (251 clips) is
reported but never fed back into training.

Run:
    .venv/bin/python -m src.vision.train_smile_pd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]

NEGATIVE_LABELS = {"no", "n", "0"}


def _binarize_pd(value, nan_as: str) -> int | None:
    """Binary label mapping.

    UFNet's exact rule (unimodal_smile_baal.py:130):
        `0 if str(x) in ['no','0'] else 1` — anything not-literally-"no"/"0",
        including NaN and "Possible"/"Probable", becomes PD=1.

    We expose a CLI switch so the reader can see both behaviours:
      - nan_as="pd"   → UFNet's exact rule (NaN → 1). Directly comparable to their AUROC.
      - nan_as="drop" → NaN excluded from training/eval. Cleaner but not what UFNet did.

    "Possible"/"Probable" are always mapped to 1 (both settings), which matches UFNet
    and is clinically defensible.
    """
    s = str(value).strip().lower()
    if s in NEGATIVE_LABELS:
        return 0
    if s in {"nan", "", "none"}:
        return 1 if nan_as == "pd" else None
    return 1


def _load_participant_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("smile_")]


# Named feature-set filters for ablation. Each predicate takes a smile_* column name
# and returns True to keep. `None` means "keep all 42".
GEOMETRIC_TOKENS = ("eye-open", "eye-raise", "mouth-open", "mouth-width", "jaw-open")
SMILE_RELATED_AUS = ("AU06", "AU12", "AU14", "AU25", "AU26")  # drop AU01 (brow) + AU45 (blink)

FEATURE_SETS: dict[str, callable | None] = {
    "all": None,
    "no_au45": lambda c: "AU45" not in c,
    "no_entropy": lambda c: not c.endswith("_entropy"),
    "au_only": lambda c: not any(t in c for t in GEOMETRIC_TOKENS),
    "au_mean_var": lambda c: (
        not any(t in c for t in GEOMETRIC_TOKENS)
        and not c.endswith("_entropy")
    ),
    "au_mean_var_no45": lambda c: (
        not any(t in c for t in GEOMETRIC_TOKENS)
        and not c.endswith("_entropy")
        and "AU45" not in c
    ),
    "au5_mean_var": lambda c: (
        any(au in c for au in SMILE_RELATED_AUS)
        and not c.endswith("_entropy")
    ),
}


def _load_training_csv(csv_path: Path, nan_as: str):
    df = pd.read_csv(csv_path)
    feature_cols = _feature_columns(df)
    assert len(feature_cols) == 42, f"Expected 42 smile_* columns, got {len(feature_cols)}"
    df[feature_cols] = df[feature_cols].fillna(0)
    df["label"] = df["pd"].map(lambda v: _binarize_pd(v, nan_as=nan_as))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df, feature_cols


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _print_metrics(name: str, m: dict) -> None:
    print(
        f"[{name:14s}] n={m['n']:4d} (pos={m['n_pos']:3d}/neg={m['n_neg']:3d})  "
        f"AUROC={m['auroc']:.3f}  AP={m['ap']:.3f}  F1={m['f1']:.3f}  "
        f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}"
    )


def _split_by_participant(df: pd.DataFrame, dev_ids: set[str], test_ids: set[str]):
    # UFNet's split files list values from the `ID` column (see unimodal_smile_baal.py:136
    # `IDs = df['ID']`), NOT the `Participant_ID` column. Using Participant_ID here would
    # only match ~43% of test IDs and silently leak most held-out subjects into train.
    ids = df["ID"].astype(str)
    is_dev = ids.isin(dev_ids)
    is_test = ids.isin(test_ids)
    is_train = ~(is_dev | is_test)
    return df[is_train], df[is_dev], df[is_test]


def train(
    paths_yaml: Path = REPO_ROOT / "configs" / "paths.yaml",
    seed: int = 42,
    nan_as: str = "pd",
    save: bool = True,
    feature_set: str = "all",
) -> dict:
    with open(paths_yaml) as f:
        paths = yaml.safe_load(f)

    csv_path = REPO_ROOT / paths["ufnet_smile_csv"]
    yt_csv_path = REPO_ROOT / paths["ufnet_smile_youtube_pd_csv"]
    splits_dir = REPO_ROOT / paths["ufnet_smile_splits_dir"]
    classifier_out = REPO_ROOT / paths["smile_pd_classifier"]
    scaler_out = REPO_ROOT / paths["smile_pd_scaler"]
    columns_out = classifier_out.with_name("smile_pd_columns.json")
    metrics_out = classifier_out.with_name("smile_pd_metrics.json")
    classifier_out.parent.mkdir(parents=True, exist_ok=True)

    df, all_feature_cols = _load_training_csv(csv_path, nan_as=nan_as)
    predicate = FEATURE_SETS[feature_set]
    feature_cols = all_feature_cols if predicate is None else [c for c in all_feature_cols if predicate(c)]
    dev_ids = _load_participant_ids(splits_dir / "dev.txt")
    test_ids = _load_participant_ids(splits_dir / "test.txt")

    train_df, dev_df, test_df = _split_by_participant(df, dev_ids, test_ids)
    print(f"[nan_as={nan_as} | feature_set={feature_set} | n_features={len(feature_cols)}]")
    print(
        f"Loaded {len(df)} sessions from {df['Participant_ID'].nunique()} participants "
        f"(pos={int((df.label==1).sum())} / neg={int((df.label==0).sum())})"
    )
    print(
        f"Split: train={len(train_df)} ({train_df['ID'].nunique()} unique IDs, "
        f"{train_df['Participant_ID'].nunique()} participants) | "
        f"dev={len(dev_df)} ({dev_df['ID'].nunique()} IDs) | "
        f"test={len(test_df)} ({test_df['ID'].nunique()} IDs)"
    )

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_dev = dev_df[feature_cols].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    X_train_res, y_train_res = SMOTE(random_state=seed).fit_resample(X_train_s, y_train)
    print(
        f"After SMOTE: train n={len(y_train_res)} "
        f"(pos={int((y_train_res==1).sum())} / neg={int((y_train_res==0).sum())})"
    )

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X_train_res, y_train_res)

    train_score = clf.predict_proba(X_train_s)[:, 1]
    dev_score = clf.predict_proba(X_dev_s)[:, 1]
    test_score = clf.predict_proba(X_test_s)[:, 1]

    print()
    train_metrics = _metrics(y_train, train_score)
    dev_metrics = _metrics(y_dev, dev_score)
    test_metrics = _metrics(y_test, test_score)
    _print_metrics("train (fit)", train_metrics)
    _print_metrics("dev", dev_metrics)
    _print_metrics("test", test_metrics)

    yt_metrics = None
    if yt_csv_path.exists():
        yt_df = pd.read_csv(yt_csv_path)
        yt_df[feature_cols] = yt_df[feature_cols].fillna(0)
        yt_df["label"] = yt_df["pd"].map(lambda v: _binarize_pd(v, nan_as=nan_as))
        yt_df = yt_df.dropna(subset=["label"])
        yt_df["label"] = yt_df["label"].astype(int)
        # YoutubePD CSV lacks the ID/Participant_ID/date metadata columns of the
        # primary CSV, but the 42 smile_* columns are named identically.
        X_yt = yt_df[feature_cols].to_numpy()
        y_yt = yt_df["label"].to_numpy()
        X_yt_s = scaler.transform(X_yt)
        yt_score = clf.predict_proba(X_yt_s)[:, 1]
        yt_metrics = _metrics(y_yt, yt_score)
        _print_metrics("youtube-pd", yt_metrics)

    result = {
        "seed": seed,
        "nan_as": nan_as,
        "n_features": len(feature_cols),
        "train": train_metrics,
        "dev": dev_metrics,
        "test": test_metrics,
        "youtube_pd": yt_metrics,
    }
    if save:
        joblib.dump(clf, classifier_out)
        joblib.dump(scaler, scaler_out)
        columns_out.write_text(json.dumps(feature_cols, indent=2))
        metrics_out.write_text(json.dumps(result, indent=2))
        print()
        print(f"Saved classifier -> {classifier_out.relative_to(REPO_ROOT)}")
        print(f"Saved scaler     -> {scaler_out.relative_to(REPO_ROOT)}")
        print(f"Saved columns    -> {columns_out.relative_to(REPO_ROOT)}")
        print(f"Saved metrics    -> {metrics_out.relative_to(REPO_ROOT)}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--nan-as",
        choices=["pd", "drop"],
        default="pd",
        help="How to treat rows where the `pd` column is NaN. "
             "'pd' = UFNet's exact rule (NaN -> PD=1, directly comparable to their AUROC). "
             "'drop' = exclude NaN rows (cleaner but deviates from UFNet).",
    )
    p.add_argument("--no-save", action="store_true", help="Skip saving joblib artifacts (useful for A/B runs).")
    p.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS.keys()),
        default="all",
        help="Column filter for ablation. Default 'all' = UFNet's 42-feature reference set.",
    )
    args = p.parse_args()
    train(seed=args.seed, nan_as=args.nan_as, save=not args.no_save, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
