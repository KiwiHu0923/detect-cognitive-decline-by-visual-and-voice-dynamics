"""DDK / articulation features from the PATAKA (/pa-ta-ka/) task.

Intensity-envelope peak-picking — no ASR, language-neutral. Eight features
per file: peak count, duration, DDK rate, ISI mean/CV (timing regularity),
peak-amplitude mean/CV (amplitude regularity), and a linear-fit amplitude
decrement (fatigue / decrescendo). Batch mode runs over the PATAKA file for
each subject in the analysis cohort; per-subject output equals per-file
(one PATAKA per subject).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import yaml
from scipy.signal import find_peaks

DDK_TASK = "PATAKA"

FEATURE_NAMES = [
    "n_peaks",
    "duration_s",
    "ddk_rate_hz",
    "isi_mean_s",
    "isi_cv",
    "amp_mean",
    "amp_cv",
    "amp_decrement",
]

logger = logging.getLogger(__name__)


def _default_cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((repo_root / "configs/model.yaml").read_text())["ddk"]


def _envelope(sound: parselmouth.Sound, smoothing_ms: float) -> tuple[np.ndarray, float]:
    """Rectify then moving-average smooth. Returns (envelope, fs)."""
    x = sound.values[0]
    fs = float(sound.sampling_frequency)
    rect = np.abs(x)
    win = max(1, int(fs * smoothing_ms / 1000))
    kernel = np.ones(win) / win
    env = np.convolve(rect, kernel, mode="same")
    return env, fs


def extract_ddk_features(
    wav_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
    cfg: dict | None = None,
) -> dict:
    """Return {FEATURE_NAMES: float} for one PATAKA WAV, or {} on failure.

    ``amp_decrement`` is set to NaN if fewer than
    ``cfg['amplitude_decrement_min_peaks']`` peaks are detected (linreg on
    too few points is noise). If fewer than 3 peaks total, the file is
    unusable and {} is returned.
    """
    if cfg is None:
        cfg = _default_cfg()

    try:
        snd = parselmouth.Sound(str(wav_path))
        if t0 is not None and t1 is not None:
            snd = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)

        env, fs = _envelope(snd, cfg["envelope_smoothing_ms"])
        env_max = float(env.max())
        if env_max <= 0:
            logger.warning("skip %s: silent envelope", wav_path)
            return {}

        peaks, _ = find_peaks(
            env,
            prominence=cfg["peak_min_prominence"] * env_max,
            distance=max(1, int(cfg["peak_min_isi_s"] * fs)),
        )
        n = len(peaks)
        if n < 3:
            logger.warning("skip %s: only %d peaks detected", wav_path, n)
            return {}

        duration = float(snd.duration)
        peak_times = peaks.astype(float) / fs
        peak_amps = env[peaks]
        isi = np.diff(peak_times)

        isi_mean = float(isi.mean())
        isi_cv = float(isi.std() / isi_mean) if isi_mean > 0 else float("nan")
        amp_mean = float(peak_amps.mean())
        amp_cv = float(peak_amps.std() / amp_mean) if amp_mean > 0 else float("nan")

        if n >= cfg["amplitude_decrement_min_peaks"]:
            slope = float(np.polyfit(peak_times, peak_amps, 1)[0])
        else:
            slope = float("nan")

        return {
            "n_peaks": int(n),
            "duration_s": duration,
            "ddk_rate_hz": float(n / duration),
            "isi_mean_s": isi_mean,
            "isi_cv": isi_cv,
            "amp_mean": amp_mean,
            "amp_cv": amp_cv,
            "amp_decrement": slope,
        }
    except Exception as e:
        logger.warning("ddk extraction failed on %s: %s", wav_path, e)
        return {}


def batch_extract(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    task: str = DDK_TASK,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Extract DDK features for every analysis-cohort subject's ``task`` file.

    Returns one row per subject (there is one PATAKA per subject by cohort
    construction). Same schema is used for both per_file and per_subject
    outputs downstream, to keep parity with the phonation pipeline.
    """
    if cfg is None:
        cfg = _default_cfg()

    in_cohort = set(cohort_df[cohort_df["in_analysis_cohort"]]["subject_id"])
    sel = labels_df[
        labels_df["subject_id"].isin(in_cohort) & (labels_df["task"] == task)
    ]

    repo_root = Path(__file__).resolve().parents[2]
    records: list[dict] = []
    for row in sel.itertuples(index=False):
        feats = extract_ddk_features(repo_root / row.file_path, cfg=cfg)
        if not feats:
            continue
        feats["subject_id"] = row.subject_id
        feats["task"] = row.task
        feats["group"] = row.group
        records.append(feats)

    df = pd.DataFrame(records)
    return df[["subject_id", "group", "task"] + FEATURE_NAMES]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())

    labels = pd.read_csv(repo_root / paths["neurovoz_labels"])
    cohort = pd.read_csv(repo_root / paths["cohort_csv"])

    df = batch_extract(labels, cohort)

    out_dir = repo_root / paths["ddk_features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file.csv"
    per_subject_path = out_dir / "per_subject.csv"
    df.to_csv(per_file_path, index=False)
    # One PATAKA per subject → per_subject = per_file with task column dropped
    df.drop(columns=["task"]).to_csv(per_subject_path, index=False)

    print(f"Per-file: {len(df)} rows → {per_file_path.relative_to(repo_root)}")
    print(
        f"Per-subject: {len(df)} rows → {per_subject_path.relative_to(repo_root)}"
    )
    n_pd = int((df["group"] == "PD").sum())
    n_hc = int((df["group"] == "HC").sum())
    print(f"Subjects with features: {n_pd} PD, {n_hc} HC (target: 49 PD, 46 HC)")

    print("\nPer-group means (expect: PD lower rate, higher ISI CV, higher amp CV):")
    means = df.groupby("group")[FEATURE_NAMES].mean().round(4)
    print(means.T.to_string())


if __name__ == "__main__":
    main()
