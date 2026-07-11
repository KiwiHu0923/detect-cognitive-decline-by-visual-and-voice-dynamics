"""End-to-end smile-based PD probability from a video clip.

Pipeline: video → OpenFace (per-frame AU_r + AU_c) → active-frame-only mean+var
(14-dim vector) → StandardScaler → LogReg → PD probability.

Refuses to emit a score when the OpenFace face-detection rate falls below
`min_detection_rate` (default 0.80). Dropped frames are not random — they
correlate with strong head motion, occlusion, and facial-expression peaks — so a
low detection rate systematically biases the sample toward *less* expressive
frames, mimicking hypomimia. Returning `score=None` with a warning is honest.

CLI: `python -m src.vision.predict_smile_pd <video> [--t0 X --t1 Y]`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.vision.aggregate import aggregate
from src.vision.facial_features import extract_smile_features
from src.vision.summarize import summarize_facial_features

REPO_ROOT = Path(__file__).resolve().parents[2]

# If this many output columns are zero-filled (i.e. ≥3 AUs never fired), the
# subject probably didn't perform the smile task properly. Flag it.
ZERO_FILL_WARNING_THRESHOLD = 6


def _load_artifacts(paths_yaml: Path = REPO_ROOT / "configs" / "paths.yaml"):
    with open(paths_yaml) as f:
        paths = yaml.safe_load(f)
    classifier_path = REPO_ROOT / paths["smile_pd_classifier"]
    scaler_path = REPO_ROOT / paths["smile_pd_scaler"]
    columns_path = classifier_path.with_name("smile_pd_columns.json")
    for p in (classifier_path, scaler_path, columns_path):
        if not p.exists():
            raise FileNotFoundError(
                f"artifact missing: {p.relative_to(REPO_ROOT)} — "
                f"run `python -m src.vision.train_smile_pd --feature-set au_mean_var` first"
            )
    clf = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    columns: list[str] = json.loads(columns_path.read_text())
    return clf, scaler, columns


def predict(
    video_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
    min_detection_rate: float = 0.80,
) -> dict:
    """Score a video and return a structured result.

    Returns dict with keys:
      score                 PD probability in [0, 1], or None if quality gate failed
      detection_rate        fraction of frames with confidence >= 0.75 and success == 1
      quality_gate_pass     detection_rate >= min_detection_rate
      features              {column_name: raw_value} for all 14 output columns
      per_au_active_frames  {AU code: count of active frames}
      zero_filled_columns   list of columns zero-filled because the AU never fired
      warnings              list of human-readable warnings from any stage
    """
    clf, scaler, columns = _load_artifacts()

    arrays, feat_meta = extract_smile_features(video_path, t0=t0, t1=t1)
    warnings = list(feat_meta["warnings"])

    if not feat_meta["quality_gate_pass"] or feat_meta["detection_rate"] < min_detection_rate:
        # Below quality gate — refuse to score.
        return {
            "score": None,
            "detection_rate": feat_meta["detection_rate"],
            "quality_gate_pass": False,
            "features": {},
            "per_au_active_frames": {},
            "zero_filled_columns": [],
            "warnings": warnings + [
                f"score withheld: detection rate {feat_meta['detection_rate']:.2%} "
                f"below {min_detection_rate:.0%} threshold"
            ],
        }

    vector, agg_meta = aggregate(arrays, columns)
    if len(agg_meta["zero_filled_columns"]) >= ZERO_FILL_WARNING_THRESHOLD:
        n_dead_aus = len(agg_meta["zero_filled_columns"]) // 2
        warnings.append(
            f"{n_dead_aus} of 7 AUs never activated in the clip — the smile task "
            f"may not have been performed as instructed (protocol: smile ×3 alternating "
            f"with neutral, 8–12 seconds total)"
        )

    X = scaler.transform(vector.reshape(1, -1))
    score = float(clf.predict_proba(X)[0, 1])

    return {
        "score": score,
        "detection_rate": feat_meta["detection_rate"],
        "quality_gate_pass": True,
        "features": {col: float(v) for col, v in zip(columns, vector)},
        "per_au_active_frames": agg_meta["per_au_active_frames"],
        "zero_filled_columns": agg_meta["zero_filled_columns"],
        "warnings": warnings,
    }


def predict_and_summarize(
    video_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
    min_detection_rate: float = 0.80,
) -> dict:
    """One OpenFace Docker run → classifier score + hypomimia narrative summary.

    Pipeline uses this on the demo path so a single Docker invocation feeds
    both the smile-PD classifier (arrays) and ``summarize_facial_features``
    (raw DataFrame) — the "one OpenFace run per demo" invariant from
    CLAUDE.md Facial Features. Semantics of the classifier fields match
    ``predict()`` exactly; ``summary`` is the same JSON schema as calling
    ``summarize_facial_features`` on the saved CSV, so downstream consumers
    (llm_fusion.build_claude_context) do not care which entry point produced it.
    """
    clf, scaler, columns = _load_artifacts()

    arrays, feat_meta, df = extract_smile_features(
        video_path, t0=t0, t1=t1, return_dataframe=True
    )
    warnings = list(feat_meta["warnings"])

    # Summary is computed on the same DataFrame regardless of the quality gate
    # — it's narrative colour for the Claude report, useful even when the
    # classifier withholds a numeric score (Claude can still say "smile
    # amplitude was low" even if we refuse to quote a PD probability).
    summary = summarize_facial_features(df)

    if not feat_meta["quality_gate_pass"] or feat_meta["detection_rate"] < min_detection_rate:
        return {
            "score": None,
            "detection_rate": feat_meta["detection_rate"],
            "quality_gate_pass": False,
            "features": {},
            "per_au_active_frames": {},
            "zero_filled_columns": [],
            "warnings": warnings + [
                f"score withheld: detection rate {feat_meta['detection_rate']:.2%} "
                f"below {min_detection_rate:.0%} threshold"
            ],
            "summary": summary,
        }

    vector, agg_meta = aggregate(arrays, columns)
    if len(agg_meta["zero_filled_columns"]) >= ZERO_FILL_WARNING_THRESHOLD:
        n_dead_aus = len(agg_meta["zero_filled_columns"]) // 2
        warnings.append(
            f"{n_dead_aus} of 7 AUs never activated in the clip — the smile task "
            f"may not have been performed as instructed (protocol: smile ×3 alternating "
            f"with neutral, 8–12 seconds total)"
        )

    X = scaler.transform(vector.reshape(1, -1))
    score = float(clf.predict_proba(X)[0, 1])

    return {
        "score": score,
        "detection_rate": feat_meta["detection_rate"],
        "quality_gate_pass": True,
        "features": {col: float(v) for col, v in zip(columns, vector)},
        "per_au_active_frames": agg_meta["per_au_active_frames"],
        "zero_filled_columns": agg_meta["zero_filled_columns"],
        "warnings": warnings,
        "summary": summary,
    }


def _explain_top_contributors(clf, scaler, features: dict[str, float], columns: list[str], k: int = 3) -> tuple[list, list]:
    """Return (top_k_toward_PD, top_k_toward_healthy) as [(column, contribution), ...]."""
    vec = np.array([features[c] for c in columns]).reshape(1, -1)
    scaled = scaler.transform(vec)[0]
    contributions = clf.coef_[0] * scaled
    order = np.argsort(-contributions)
    toward_pd = [(columns[i], float(contributions[i])) for i in order[:k] if contributions[i] > 0]
    toward_healthy = [(columns[i], float(contributions[i])) for i in order[::-1][:k] if contributions[i] < 0]
    return toward_pd, toward_healthy


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Path to a video containing the smile task.")
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--t1", type=float, default=None)
    p.add_argument("--min-detection-rate", type=float, default=0.80)
    args = p.parse_args()

    result = predict(
        args.video,
        t0=args.t0,
        t1=args.t1,
        min_detection_rate=args.min_detection_rate,
    )

    print(f"detection rate:    {result['detection_rate']:.2%}")
    print(f"quality gate pass: {result['quality_gate_pass']}")
    if result["score"] is None:
        print("PD score:          — (withheld)")
    else:
        print(f"PD score:          {result['score']:.3f}")
    print()
    if result["per_au_active_frames"]:
        print("per-AU active frames:")
        for au, n in result["per_au_active_frames"].items():
            marker = "  (never fired)" if n == 0 else ""
            print(f"  {au}: {n}{marker}")
    if result["zero_filled_columns"]:
        print(f"\nzero-filled columns ({len(result['zero_filled_columns'])}): "
              f"{result['zero_filled_columns']}")
    for w in result["warnings"]:
        print(f"⚠ {w}")

    if result["score"] is not None:
        clf, scaler, columns = _load_artifacts()
        toward_pd, toward_healthy = _explain_top_contributors(
            clf, scaler, result["features"], columns
        )
        print("\ntop contributors toward PD:")
        for col, c in toward_pd:
            print(f"  +{c:.3f}  {col} = {result['features'][col]:.3f}")
        print("top contributors toward healthy:")
        for col, c in toward_healthy:
            print(f"  {c:+.3f}  {col} = {result['features'][col]:.3f}")


if __name__ == "__main__":
    _main()
