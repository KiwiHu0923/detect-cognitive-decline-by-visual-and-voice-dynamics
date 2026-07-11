"""Distill an OpenFace CSV into a JSON summary of hypomimia markers.

Reads the same CSV that `facial_features.extract_smile_features` already asked
OpenFace to produce — no second Docker run, no py-feat dependency. The output
is narrative colour for the Claude report layer, not a classifier: it names the
clinical hypomimia markers that Claude can talk about, using the same
per-frame data the smile classifier consumed for its score.

Design decisions worth preserving:

  * `AU12_amplitude_on_smile_cue` is the **max** of AU12_r on active frames
    (frames with `AU12_c == 1`), not the active-frame mean. Rationale:
      (a) Clinical "smile amplitude" in hypomimia literature refers to the
          peak lip-corner raise a patient can produce, not the average.
      (b) The smile classifier's 14-dim feature vector already includes
          active-frame mean for AU12; passing peak here gives Claude a
          non-redundant number.
    Two filters already protect against single-frame noise:
      - OpenFace confidence < 0.75 → dropped (per facial_features.py)
      - Only frames where AU12_c == 1 count → these are multi-frame smile
        episodes flagged by OpenFace, not random one-frame spikes.

  * `expression_variance` is the mean of each AU's **temporal std** across all
    kept frames. This gives Claude a different view than the classifier's
    active-frame variance features — it measures how much the whole face
    moves over the clip, which is the clinical hypomimia concept ("mask-like
    face" = low overall variability). Higher = more expressive.

  * `hypomimia_score` intentionally omitted. Claude synthesises the narrative
    composite from the raw markers; adding a composite here would need an
    arbitrary normalization anchor and duplicates work Claude does anyway.

  * `blink_rate_per_min` uses AU45_c 0→1 rising edges over kept-duration in
    minutes, matching how blink rate is defined clinically (events per unit
    time, not a fraction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Match facial_features.py so summarize / classifier see the same "kept" frames.
CONFIDENCE_MIN: float = 0.75
QUALITY_GATE_MIN: float = 0.80
MIN_KEPT_FRAMES: int = 10

AUS: tuple[str, ...] = ("AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45")
POSE_AXES: tuple[str, ...] = ("pose_Tx", "pose_Ty", "pose_Tz")


def _empty_summary(detection_rate: float, warnings: list[str]) -> dict:
    return {
        "mean_AU12": None,
        "AU12_amplitude_on_smile_cue": None,
        "expression_variance": None,
        "blink_rate_per_min": None,
        "head_movement_std": None,
        "detection_rate": detection_rate,
        "warnings": warnings,
    }


def summarize_facial_features(
    openface_csv_path_or_df: str | Path | pd.DataFrame,
    min_confidence: float = CONFIDENCE_MIN,
) -> dict:
    """Compute hypomimia markers from an OpenFace CSV.

    Args:
      openface_csv_path_or_df: path to an OpenFace FeatureExtraction CSV, or
        a pre-loaded DataFrame (useful when the pipeline already has it in
        memory).
      min_confidence: face-tracking confidence floor for the "kept frames"
        mask. Match facial_features.CONFIDENCE_MIN so the classifier and
        summary see the same denominator.

    Returns a dict with:
      mean_AU12                     mean of AU12_r on all kept frames
      AU12_amplitude_on_smile_cue   max of AU12_r on active-smile frames
      expression_variance           mean of per-AU temporal std across kept frames
      blink_rate_per_min            AU45_c rising edges / kept duration
      head_movement_std             mean of std(pose_Tx/Ty/Tz) on kept frames
      detection_rate                n_kept / n_total
      warnings                      list[str] of quality flags

    Any field that can't be computed (empty CSV, no active smile frames,
    missing pose columns, etc.) is set to None with an entry in warnings.
    """
    if isinstance(openface_csv_path_or_df, pd.DataFrame):
        df = openface_csv_path_or_df.copy()
    else:
        csv_path = Path(openface_csv_path_or_df)
        if not csv_path.exists():
            raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    # OpenFace occasionally emits a leading space on column names.
    df.columns = [c.strip() for c in df.columns]

    n_total = len(df)
    if n_total == 0:
        return _empty_summary(0.0, ["OpenFace CSV is empty — no frames processed"])

    warnings: list[str] = []

    success = df["success"].to_numpy().astype(bool)
    confidence = df["confidence"].to_numpy(dtype=float)
    kept = success & (confidence >= min_confidence)
    n_kept = int(kept.sum())
    detection_rate = float(n_kept / n_total)

    if n_kept < MIN_KEPT_FRAMES:
        return _empty_summary(
            detection_rate,
            [f"insufficient face detection ({n_kept} kept frames < {MIN_KEPT_FRAMES})"],
        )
    if detection_rate < QUALITY_GATE_MIN:
        warnings.append(
            f"low face detection ({detection_rate:.1%} < {QUALITY_GATE_MIN:.0%})"
        )

    # fps from timestamps (mean rate over the clip). Matches facial_features.py.
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=float)
    else:
        ts = np.array([])
    if ts.size > 1 and ts[-1] > ts[0]:
        fps = float((n_total - 1) / (ts[-1] - ts[0]))
    else:
        fps = 0.0
        warnings.append("could not infer fps from timestamps")

    # AU12: peak amplitude on active smile frames + overall mean on kept frames.
    # Pandas 3.0's `.to_numpy()` may return a read-only Arrow view — force copy
    # before any masked in-place assignment.
    if "AU12_r" in df.columns and "AU12_c" in df.columns:
        au12_r = df["AU12_r"].to_numpy(dtype=float).copy()
        au12_c = df["AU12_c"].to_numpy(dtype=np.int8).copy()
        au12_r[~kept] = np.nan
        au12_c[~kept] = 0
        mean_au12: float | None = float(np.nanmean(au12_r))
        active_mask = au12_c == 1
        n_active_au12 = int(active_mask.sum())
        if n_active_au12 > 0:
            au12_amplitude: float | None = float(np.nanmax(au12_r[active_mask]))
        else:
            au12_amplitude = 0.0
            warnings.append("no smile detected (AU12 never active)")
    else:
        mean_au12 = None
        au12_amplitude = None
        warnings.append("missing AU12 columns")

    # Expression variance: temporal std of each AU on kept frames, averaged across AUs.
    # High = expressive face; low = mask-like (clinical hypomimia direction).
    per_au_std: list[float] = []
    for au in AUS:
        col = f"{au}_r"
        if col not in df.columns:
            warnings.append(f"missing column {col}")
            continue
        vals = df[col].to_numpy(dtype=float).copy()
        vals[~kept] = np.nan
        s = float(np.nanstd(vals))
        per_au_std.append(s)
    expression_variance: float | None = (
        float(np.mean(per_au_std)) if per_au_std else None
    )

    # Blink rate: AU45_c 0->1 rising edges per minute of kept video.
    if "AU45_c" in df.columns and fps > 0:
        au45_c = df["AU45_c"].to_numpy(dtype=np.int8).copy()
        au45_c[~kept] = 0
        diffs = np.diff(au45_c)
        rising_edges = int((diffs == 1).sum())
        kept_duration_min = n_kept / fps / 60.0
        blink_rate_per_min: float | None = (
            float(rising_edges / kept_duration_min) if kept_duration_min > 0 else None
        )
    else:
        blink_rate_per_min = None
        if "AU45_c" not in df.columns:
            warnings.append("missing AU45_c column")

    # Head movement: mean of translational std across the three axes.
    pose_stds: list[float] = []
    for axis in POSE_AXES:
        if axis not in df.columns:
            warnings.append(f"missing column {axis}")
            continue
        vals = df[axis].to_numpy(dtype=float).copy()
        vals[~kept] = np.nan
        pose_stds.append(float(np.nanstd(vals)))
    head_movement_std: float | None = (
        float(np.mean(pose_stds)) if pose_stds else None
    )

    return {
        "mean_AU12": mean_au12,
        "AU12_amplitude_on_smile_cue": au12_amplitude,
        "expression_variance": expression_variance,
        "blink_rate_per_min": blink_rate_per_min,
        "head_movement_std": head_movement_std,
        "detection_rate": detection_rate,
        "warnings": warnings,
    }


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="Path to an OpenFace FeatureExtraction CSV.")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=CONFIDENCE_MIN,
        help=f"Face-tracking confidence floor (default {CONFIDENCE_MIN}).",
    )
    args = p.parse_args()
    summary = summarize_facial_features(args.csv, min_confidence=args.min_confidence)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
