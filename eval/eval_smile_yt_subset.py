"""Re-evaluate the fitted smile classifier on UFNet's designated YouTubePD test
subset (splits/test_yt_pd.txt, 184 IDs), alongside the full 251-clip CSV.

Motivation: `youtube_PD_features.csv` has 251 rows, but UFNet ship a
`test_yt_pd.txt` split file listing only 184 IDs — we do not know why the two
disagree. Reporting both makes the difference visible.

Loads the fitted artifacts (LR + scaler + 14-column list) directly; no training
happens here. Appends a `youtube_pd_subset` key to smile_pd_metrics.json
without overwriting the existing `youtube_pd` (full 251) entry.

Run:
    .venv/bin/python -m eval.eval_smile_yt_subset
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.vision.train_smile_pd import (
    _binarize_pd,
    _load_participant_ids,
    _metrics,
    _print_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(nan_as: str = "pd") -> dict:
    with open(REPO_ROOT / "configs" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    yt_csv_path = REPO_ROOT / paths["ufnet_smile_youtube_pd_csv"]
    splits_dir = REPO_ROOT / paths["ufnet_smile_splits_dir"]
    classifier_path = REPO_ROOT / paths["smile_pd_classifier"]
    scaler_path = REPO_ROOT / paths["smile_pd_scaler"]
    columns_path = classifier_path.with_name("smile_pd_columns.json")
    metrics_path = classifier_path.with_name("smile_pd_metrics.json")

    clf = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    feature_cols = json.loads(columns_path.read_text())
    subset_ids = _load_participant_ids(splits_dir / "test_yt_pd.txt")

    yt_df = pd.read_csv(yt_csv_path)
    yt_df[feature_cols] = yt_df[feature_cols].fillna(0)
    yt_df["label"] = yt_df["pd"].map(lambda v: _binarize_pd(v, nan_as=nan_as))
    yt_df = yt_df.dropna(subset=["label"])
    yt_df["label"] = yt_df["label"].astype(int)

    def _eval(df: pd.DataFrame) -> dict:
        X = df[feature_cols].to_numpy()
        y = df["label"].to_numpy()
        score = clf.predict_proba(scaler.transform(X))[:, 1]
        return _metrics(y, score)

    full_metrics = _eval(yt_df)

    subset_df = yt_df[yt_df["ID"].astype(str).isin(subset_ids)]
    matched = int(subset_df["ID"].astype(str).nunique())
    print(
        f"[subset filter] test_yt_pd.txt lists {len(subset_ids)} IDs; "
        f"{matched} matched in youtube_PD_features.csv (dropped {len(subset_ids) - matched})"
    )
    subset_metrics = _eval(subset_df)

    _print_metrics("yt full (251)", full_metrics)
    _print_metrics("yt subset (184)", subset_metrics)

    existing = json.loads(metrics_path.read_text())
    existing["youtube_pd_subset"] = subset_metrics
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"Wrote youtube_pd_subset metrics to {metrics_path}")

    return {"full": full_metrics, "subset": subset_metrics}


if __name__ == "__main__":
    main()
