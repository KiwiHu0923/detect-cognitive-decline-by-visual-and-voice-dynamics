"""Phonation features from sustained vowels via Praat / Parselmouth.

Twelve features per WAV: 3 jitter (local, rap, ppq5), 4 shimmer (local, apq5,
apq11, dda), HNR mean (dB), F0 mean/std/range (Hz), and voicing fraction.

Batch mode (Day 4, 2026-07-10): filters to `included_vowels` (default [I, O, U]
per Li et al. arxiv 2606.19125 Table IV — [a]/[e] barely above chance and are
dropped). Aggregation is two-stage:
  1. Per-subject × per-vowel: mean across reps of the same vowel (I1/I2/I3 →
     one /i/ vector per subject).
  2. Per-subject: AUC-excess weighted average across vowels using
     `vowel_weights` from paper Table IV Person AUC.

Replaces the Day-2 pipeline whose `BALANCED_VOWEL_SET = (A1, A2, I1, O2, U1)`
double-counted /a/ (both reps were treated as separate tasks).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import yaml
from parselmouth.praat import call

INCLUDED_VOWELS_DEFAULT = ("I", "O", "U")

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


def _apply_steady_window(snd: parselmouth.Sound, cfg: dict) -> parselmouth.Sound:
    """Crop to the middle `steady_window_frac` of the sound.

    Praat jitter/shimmer/HNR assume quasi-periodic, stable-pitch signal; the
    onset/offset carry glottal transients and F0 drift that inflate the
    perturbation measures in both PD and HC files. Cropping to the steady
    middle recovers per-file discrimination without changing per-vowel identity.

    Short-clip guard: sounds below ``steady_window_min_duration_s`` are returned
    unchanged so a 60%-crop of a very short file doesn't leave < 0.3s for the
    pitch tracker.
    """
    dur = float(snd.duration)
    min_dur = float(cfg.get("steady_window_min_duration_s", 0.5))
    if dur < min_dur:
        return snd
    frac = float(cfg.get("steady_window_frac", 0.6))
    margin = (1.0 - frac) / 2.0
    return snd.extract_part(
        from_time=margin * dur,
        to_time=(1.0 - margin) * dur,
        preserve_times=False,
    )


def extract_phonation_features(
    wav_path: str | Path,
    t0: float | None = None,
    t1: float | None = None,
    cfg: dict | None = None,
) -> dict:
    """Return {FEATURE_NAMES: float} for one WAV, or {} on failure.

    Optional ``t0``/``t1`` (seconds) crop the sound before analysis — used at
    demo time after task segmentation. For the NeuroVoz batch (one task per
    file), pass the whole file. Steady-window cropping (config
    ``apply_steady_window``) runs on top of the t0/t1 crop.
    """
    if cfg is None:
        cfg = _default_cfg()

    try:
        snd = parselmouth.Sound(str(wav_path))
        if t0 is not None and t1 is not None:
            snd = snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
        if cfg.get("apply_steady_window", False):
            snd = _apply_steady_window(snd, cfg)

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


def _vowel_of(task: str) -> str:
    """First char of task ID: A1→A, I2→I, O3→O, U1→U, PATAKA→P."""
    return task[0].upper()


def apply_vowel_weights(
    per_vowel: dict[str, dict | None],
    weights: dict[str, float],
) -> dict | None:
    """Weighted average of feature vectors across vowels present for a subject.

    Weights are renormalized over the vowels that are actually present, so a
    subject missing /u/ still gets scored (I & O renorm to 0.5 / 0.5). Returns
    ``None`` if no included vowel yielded features.
    """
    present = {v: vec for v, vec in per_vowel.items() if vec is not None}
    if not present:
        return None
    w = {v: float(weights.get(v, 0.0)) for v in present}
    total = sum(w.values())
    if total <= 0.0:
        # Fallback: equal weight if no config weights match (defensive).
        w = {v: 1.0 for v in present}
        total = float(len(present))
    w_norm = {v: w[v] / total for v in present}
    return {
        c: float(sum(w_norm[v] * present[v][c] for v in present))
        for c in FEATURE_NAMES
    }


def batch_extract(
    labels_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run extraction over the analysis cohort's `included_vowels`.

    Returns ``(per_file, per_subject_per_vowel, per_subject)``:
      - ``per_file``: one row per (subject, task) file with the 12 features.
      - ``per_subject_per_vowel``: one row per (subject, vowel) — features are
        the mean across that subject's reps of that vowel (I1/I2/I3 → mean /i/).
      - ``per_subject``: one row per subject — AUC-excess weighted average
        across the vowels present, using ``cfg['vowel_weights']``.
    """
    if cfg is None:
        cfg = _default_cfg()

    included = tuple(v.upper() for v in cfg.get("included_vowels", INCLUDED_VOWELS_DEFAULT))
    weights = {k.upper(): float(v) for k, v in cfg.get("vowel_weights", {}).items()}

    in_cohort = set(cohort_df[cohort_df["in_analysis_cohort"]]["subject_id"])
    sel = labels_df[
        labels_df["subject_id"].isin(in_cohort)
        & labels_df["task"].str[0].str.upper().isin(included)
    ].copy()

    repo_root = Path(__file__).resolve().parents[2]
    records: list[dict] = []
    for row in sel.itertuples(index=False):
        feats = extract_phonation_features(repo_root / row.file_path, cfg=cfg)
        if not feats:
            continue
        feats["subject_id"] = row.subject_id
        feats["task"] = row.task
        feats["vowel"] = _vowel_of(row.task)
        feats["group"] = row.group
        records.append(feats)

    per_file = pd.DataFrame(records)
    per_file = per_file[["subject_id", "group", "task", "vowel"] + FEATURE_NAMES]

    # Per-subject × per-vowel: mean across reps of the same vowel.
    per_subject_per_vowel = (
        per_file.groupby(["subject_id", "group", "vowel"], sort=False)[FEATURE_NAMES]
        .mean()
        .reset_index()
    )
    n_reps = (
        per_file.groupby(["subject_id", "vowel"], sort=False)
        .size()
        .rename("n_reps")
        .reset_index()
    )
    per_subject_per_vowel = per_subject_per_vowel.merge(
        n_reps, on=["subject_id", "vowel"], how="left"
    )

    # Per-subject: AUC-excess weighted average across vowels present.
    per_subject_rows: list[dict] = []
    for (subject_id, group), g in per_subject_per_vowel.groupby(
        ["subject_id", "group"], sort=False
    ):
        per_vowel = {v: None for v in included}
        for row in g.itertuples(index=False):
            if row.vowel in per_vowel:
                per_vowel[row.vowel] = {c: getattr(row, c) for c in FEATURE_NAMES}
        weighted = apply_vowel_weights(per_vowel, weights)
        if weighted is None:
            continue
        vowels_used = "".join(v for v in included if per_vowel[v] is not None)
        per_subject_rows.append({
            "subject_id": subject_id,
            "group": group,
            "vowels_used": vowels_used,
            **weighted,
        })
    per_subject = pd.DataFrame(per_subject_rows)
    return per_file, per_subject_per_vowel, per_subject


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    paths = yaml.safe_load((repo_root / "configs/paths.yaml").read_text())
    cfg = _default_cfg()

    labels = pd.read_csv(repo_root / paths["neurovoz_labels"])
    cohort = pd.read_csv(repo_root / paths["cohort_csv"])

    per_file, per_subject_per_vowel, per_subject = batch_extract(labels, cohort)

    out_dir = repo_root / paths["phonation_features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file.csv"
    per_subject_per_vowel_path = out_dir / "per_subject_per_vowel.csv"
    per_subject_path = out_dir / "per_subject.csv"
    per_file.to_csv(per_file_path, index=False)
    per_subject_per_vowel.to_csv(per_subject_per_vowel_path, index=False)
    per_subject.to_csv(per_subject_path, index=False)

    print(f"Per-file: {len(per_file)} rows → {per_file_path.relative_to(repo_root)}")
    print(f"Per-subject-per-vowel: {len(per_subject_per_vowel)} rows → "
          f"{per_subject_per_vowel_path.relative_to(repo_root)}")
    print(f"Per-subject: {len(per_subject)} rows → {per_subject_path.relative_to(repo_root)}")

    included = tuple(v.upper() for v in cfg.get("included_vowels", INCLUDED_VOWELS_DEFAULT))
    weights = cfg.get("vowel_weights", {})
    print(f"\nIncluded vowels: {included}")
    print(f"Vowel weights (paper AUC-excess, arxiv 2606.19125 Table IV): {weights}")

    print("\nCoverage per vowel (subjects with ≥ 1 rep):")
    for v in included:
        v_rows = per_subject_per_vowel[per_subject_per_vowel["vowel"] == v]
        pd_n = int((v_rows["group"] == "PD").sum())
        hc_n = int((v_rows["group"] == "HC").sum())
        mean_reps = float(v_rows["n_reps"].mean()) if len(v_rows) else 0.0
        print(f"  [{v}]: PD={pd_n}, HC={hc_n}, mean reps/subject={mean_reps:.2f}")

    print("\nvowels_used pattern counts:")
    print(per_subject["vowels_used"].value_counts().to_string())

    n_pd = int((per_subject["group"] == "PD").sum())
    n_hc = int((per_subject["group"] == "HC").sum())
    print(f"\nSubjects with weighted per-subject vector: {n_pd} PD, {n_hc} HC "
          f"(target: 49 PD, 46 HC)")

    print("\nPer-group means (weighted per-subject vectors; expect PD > HC for jitter/shimmer, PD < HC for HNR):")
    means = per_subject.groupby("group")[FEATURE_NAMES].mean().round(4)
    print(means.T.to_string())


if __name__ == "__main__":
    main()
