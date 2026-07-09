"""Phonation features from sustained vowels via Praat / Parselmouth.

Twelve features per WAV: 3 jitter (local, rap, ppq5), 4 shimmer (local, apq5,
apq11, dda), HNR mean (dB), F0 mean/std/range (Hz), and voicing fraction.

Batch mode restricts to the balanced 5-vowel set (A1, A2, I1, O2, U1) — the
tasks with ≥ 44 subjects in BOTH PD and HC groups after age-matching. Per-
subject aggregation is the arithmetic mean over that subject's available
vowel files (usually 5). This is the training input for the LOSO phonation
classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import yaml
from parselmouth.praat import call

BALANCED_VOWEL_SET = ("A1", "A2", "I1", "O2", "U1")

FEATURE_NAMES = [
    "jitter_local",
    "jitter_rap",
    "jitter_ppq5",
    "shimmer_local",
    "shimmer_apq5",
    "shimmer_apq11",
    "shimmer_dda",
    "hnr_mean_db",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "voicing_fraction",
]

logger = logging.getLogger(__name__)


def _default_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((repo_root / "configs/model.yaml").read_text())["phonation"]


def extract_phonation_features(
    wav_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
    cfg: dict | None = None,
) -> dict:
    """Return {FEATURE_NAMES: float} for one WAV, or {} on failure.

    Optional ``t0``/``t1`` (seconds) crop the sound before analysis — used at
    demo time after task segmentation. For the NeuroVoz batch (one task per
    file), pass the whole file.
    """
    if cfg is None:
        cfg = _default_cfg()

    try:
        snd = parselmouth.Sound(str(wav_path))
        if t0 is not None and t1 is not None:
            snd = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)

        # Praat "To Pitch (cc)" full 10-arg signature (see empirical check).
        pitch = call(
            snd,
            "To Pitch (cc)",
            0.0,                       # time_step (auto)
            cfg["f0_floor_hz"],
            15,                        # max candidates
            0,                         # very accurate = no
            0.03,                      # silence threshold
            0.45,                      # voicing threshold
            0.01,                      # octave cost
            0.35,                      # octave-jump cost
            0.14,                      # voiced/unvoiced cost
            cfg["f0_ceiling_hz"],
        )

        f0 = pitch.selected_array["frequency"]
        if len(f0) == 0:
            logger.warning("skip %s: empty pitch track", wav_path)
            return {}
        voiced = f0[f0 > 0]
        voicing_frac = float(len(voiced) / len(f0))
        voiced_seconds = voicing_frac * float(snd.duration)
        if voiced_seconds < cfg["min_voicing_seconds"]:
            logger.warning(
                "skip %s: only %.2fs voiced (< %.1fs)",
                wav_path,
                voiced_seconds,
                cfg["min_voicing_seconds"],
            )
            return {}

        pp = call([snd, pitch], "To PointProcess (cc)")

        jf = cfg["jitter_period_floor_s"]
        jc = cfg["jitter_period_ceiling_s"]
        pmax = cfg["jitter_max_period_factor"]
        amax = cfg["jitter_max_amplitude_factor"]

        j_local = call(pp, "Get jitter (local)", 0, 0, jf, jc, pmax)
        j_rap = call(pp, "Get jitter (rap)", 0, 0, jf, jc, pmax)
        j_ppq5 = call(pp, "Get jitter (ppq5)", 0, 0, jf, jc, pmax)
        s_local = call([snd, pp], "Get shimmer (local)", 0, 0, jf, jc, pmax, amax)
        s_apq5 = call([snd, pp], "Get shimmer (apq5)", 0, 0, jf, jc, pmax, amax)
        s_apq11 = call([snd, pp], "Get shimmer (apq11)", 0, 0, jf, jc, pmax, amax)
        s_dda = call([snd, pp], "Get shimmer (dda)", 0, 0, jf, jc, pmax, amax)

        harm = call(snd, "To Harmonicity (cc)", 0.01, cfg["f0_floor_hz"], 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)

        return {
            "jitter_local": float(j_local),
            "jitter_rap": float(j_rap),
            "jitter_ppq5": float(j_ppq5),
            "shimmer_local": float(s_local),
            "shimmer_apq5": float(s_apq5),
            "shimmer_apq11": float(s_apq11),
            "shimmer_dda": float(s_dda),
            "hnr_mean_db": float(hnr),
            "f0_mean_hz": float(voiced.mean()),
            "f0_std_hz": float(voiced.std()),
            "f0_range_hz": float(voiced.max() - voiced.min()),
            "voicing_fraction": voicing_frac,
        }
    except Exception as e:
        logger.warning("phonation extraction failed on %s: %s", wav_path, e)
        return {}


def batch_extract(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    vowel_set: tuple[str, ...] = BALANCED_VOWEL_SET,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run extraction over the analysis cohort's balanced vowel set.

    Returns ``(per_file, per_subject)`` — per_subject is the arithmetic mean of
    per_file over each subject's available files.
    """
    if cfg is None:
        cfg = _default_cfg()

    in_cohort = set(cohort_df[cohort_df["in_analysis_cohort"]]["subject_id"])
    sel = labels_df[
        labels_df["subject_id"].isin(in_cohort) & labels_df["task"].isin(vowel_set)
    ].copy()

    repo_root = Path(__file__).resolve().parents[2]
    records: list[dict] = []
    for row in sel.itertuples(index=False):
        feats = extract_phonation_features(repo_root / row.file_path, cfg=cfg)
        if not feats:
            continue
        feats["subject_id"] = row.subject_id
        feats["task"] = row.task
        feats["group"] = row.group
        records.append(feats)

    per_file = pd.DataFrame(records)
    per_file = per_file[["subject_id", "group", "task"] + FEATURE_NAMES]

    per_subject = (
        per_file.groupby(["subject_id", "group"], sort=False)[FEATURE_NAMES]
        .mean()
        .reset_index()
    )
    return per_file, per_subject


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())

    labels = pd.read_csv(repo_root / paths["neurovoz_labels"])
    cohort = pd.read_csv(repo_root / paths["cohort_csv"])

    per_file, per_subject = batch_extract(labels, cohort)

    out_dir = repo_root / paths["phonation_features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file.csv"
    per_subject_path = out_dir / "per_subject.csv"
    per_file.to_csv(per_file_path, index=False)
    per_subject.to_csv(per_subject_path, index=False)

    print(f"Per-file: {len(per_file)} rows → {per_file_path.relative_to(repo_root)}")
    print(
        f"Per-subject: {len(per_subject)} rows → {per_subject_path.relative_to(repo_root)}"
    )

    n_pd = int((per_subject["group"] == "PD").sum())
    n_hc = int((per_subject["group"] == "HC").sum())
    print(f"Subjects with features: {n_pd} PD, {n_hc} HC (target: 49 PD, 46 HC)")

    print("\nPer-group means (should show PD > HC for jitter/shimmer, PD < HC for HNR):")
    means = per_subject.groupby("group")[FEATURE_NAMES].mean().round(4)
    print(means.T.to_string())


if __name__ == "__main__":
    main()
