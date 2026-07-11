"""Quick multimodal PD score for a single subject — skips the LLM/Claude layer.

Feeds phonation + DDK + facial classifiers with the three task-matched
recordings for one subject; prints per-channel PD probability, AUC-excess
weighted fusion score, and predicted class. Sanity check for the multimodal
pipeline before the Claude report layer lands.

Any of the three dirs may be omitted → that channel returns N/A and fusion
weights renormalize over the present channels (matches `llm_fusion.py`
convention).

Usage:
  python -m src.fusion.quick_score \
    --vowel-dir data/samples/pd_demo/vowel \
    --pataka-dir data/samples/pd_demo/pataka \
    --smile-dir data/samples/pd_demo/smile \
    [--label PD]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.audio.ddk import extract_ddk_features
from src.audio.phonation import (
    INCLUDED_VOWELS_DEFAULT,
    apply_vowel_weights,
    extract_phonation_features,
)
from src.fusion.llm_fusion import fuse_scores
from src.vision.predict_smile_pd import predict as predict_smile

REPO_ROOT = Path(__file__).resolve().parents[2]

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".mp4", ".mov", ".flac", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _to_wav(src: Path) -> tuple[Path, bool]:
    """Return (wav_path, is_temp). Converts non-WAV to 16 kHz mono temp WAV via ffmpeg."""
    if src.suffix.lower() == ".wav":
        return src, False
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True, capture_output=True,
    )
    return tmp, True


def _aggregate_features(per_file: list[dict], feature_cols: list[str]) -> dict | None:
    """Mean across files, per-feature nanmean. Returns None if any column is all-NaN."""
    out = {}
    for c in feature_cols:
        arr = np.array([f.get(c, np.nan) for f in per_file], dtype=float)
        if np.all(np.isnan(arr)):
            return None
        out[c] = float(np.nanmean(arr))
    return out


def _vowel_of_filename(name: str) -> str | None:
    """Extract vowel letter from filenames like HC_A1.m4a, PD_U2.m4a → 'A', 'U'.

    Assumes the underscore-delimited last token starts with the vowel letter
    followed by the rep number. Returns None if parsing fails.
    """
    stem = Path(name).stem
    tokens = stem.split("_")
    if not tokens or not tokens[-1]:
        return None
    return tokens[-1][0].upper()


def _score_phonation(vowel_dir: Path) -> tuple[float | None, dict]:
    """Per-vowel rep averaging + AUC-excess weighted aggregation.

    Mirrors training pipeline (Day 4, 2026-07-10): (1) group demo vowel files
    by vowel letter parsed from the filename, (2) skip vowels not in
    ``included_vowels`` config, (3) average features across reps of each vowel,
    (4) apply paper AUC-excess weights across vowels present. Filenames must
    follow ``<group>_<VOWEL><REP>.<ext>`` (e.g. HC_A1.m4a, PD_U2.m4a).
    """
    cfg_all = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    phon_cfg = cfg_all["phonation"]
    included = tuple(v.upper() for v in phon_cfg.get("included_vowels", INCLUDED_VOWELS_DEFAULT))
    weights = {k.upper(): float(v) for k, v in phon_cfg.get("vowel_weights", {}).items()}

    clf = joblib.load(REPO_ROOT / "eval/models/phonation.joblib")
    meta = json.loads((REPO_ROOT / "eval/models/phonation_meta.json").read_text())
    cols = meta["feature_cols"]

    per_vowel_feats: dict[str, list[dict]] = {v: [] for v in included}
    used_names: dict[str, list[str]] = {v: [] for v in included}
    skipped: list[dict] = []
    tmps: list[Path] = []

    for f in sorted(vowel_dir.iterdir()):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        v = _vowel_of_filename(f.name)
        if v is None:
            skipped.append({"file": f.name, "reason": "unparseable vowel"})
            continue
        if v not in included:
            skipped.append({"file": f.name, "reason": f"vowel [{v}] not in included_vowels {included}"})
            continue
        wav, is_tmp = _to_wav(f)
        if is_tmp:
            tmps.append(wav)
        feats = extract_phonation_features(wav)
        if not feats:
            skipped.append({"file": f.name, "reason": "extraction failed"})
            continue
        per_vowel_feats[v].append(feats)
        used_names[v].append(f.name)
    for t in tmps:
        t.unlink(missing_ok=True)

    # Per-vowel: mean of reps.
    per_vowel_agg: dict[str, dict | None] = {}
    for v in included:
        reps = per_vowel_feats[v]
        if not reps:
            per_vowel_agg[v] = None
            continue
        per_vowel_agg[v] = {c: float(np.nanmean([r[c] for r in reps])) for c in cols}

    weighted = apply_vowel_weights(per_vowel_agg, weights)
    if weighted is None:
        return None, {
            "reason": "no included-vowel files available",
            "skipped": skipped,
            "included_vowels": list(included),
        }

    X = np.array([[weighted[c] for c in cols]])
    prob = float(clf.predict_proba(X)[0, 1])
    vowels_present = [v for v in included if per_vowel_agg[v] is not None]
    return prob, {
        "n_files_used": sum(len(used_names[v]) for v in included),
        "n_files_skipped": len(skipped),
        "vowels_present": vowels_present,
        "reps_per_vowel": {v: len(used_names[v]) for v in vowels_present},
        "files_by_vowel": {v: used_names[v] for v in vowels_present},
        "skipped": skipped,
        "features": weighted,
    }


def _score_ddk(pataka_dir: Path) -> tuple[float | None, dict]:
    clf = joblib.load(REPO_ROOT / "eval/models/ddk.joblib")
    meta = json.loads((REPO_ROOT / "eval/models/ddk_meta.json").read_text())
    cols = meta["feature_cols"]

    per_file, per_file_names, tmps = [], [], []
    for f in sorted(pataka_dir.iterdir()):
        if f.suffix.lower() not in AUDIO_EXTS:
            continue
        wav, is_tmp = _to_wav(f)
        if is_tmp:
            tmps.append(wav)
        feats = extract_ddk_features(wav)
        if feats:
            per_file.append(feats)
            per_file_names.append(f.name)
    for t in tmps:
        t.unlink(missing_ok=True)

    if not per_file:
        return None, {"reason": "no PATAKA files yielded features"}

    mean_feats = _aggregate_features(per_file, cols)
    if mean_feats is None:
        return None, {"reason": "aggregated feature vector had all-NaN column"}

    X = np.array([[mean_feats[c] for c in cols]])
    prob = float(clf.predict_proba(X)[0, 1])
    return prob, {"n_files": len(per_file), "files": per_file_names, "features": mean_feats}


def _score_facial(smile_dir: Path) -> tuple[float | None, dict]:
    per_clip = []
    scored_clips = []  # (score, file_name) for clips that passed the gate
    for f in sorted(smile_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        r = predict_smile(f)
        per_clip.append({
            "file": f.name,
            "score": r["score"],
            "detection_rate": r["detection_rate"],
            "warnings": r["warnings"],
        })
        if r["score"] is not None:
            scored_clips.append((r["score"], f.name))
    if not scored_clips:
        return None, {"reason": "all smile clips withheld by quality gate", "per_clip": per_clip}
    # Optimistic max-pool: pick the clip most suggestive of PD. Rationale: a
    # single genuine hypomimia flash is diagnostically informative; averaging
    # dilutes it against clips where the subject happens to smile normally.
    # Trade-off documented — an unusually flat neutral pose on HC input will
    # also boost this score, so HC calibration needs its own eyeball check.
    best_score, best_file = max(scored_clips, key=lambda t: t[0])
    return float(best_score), {
        "n_clips_used": len(scored_clips),
        "policy": "max",
        "selected_clip": best_file,
        "per_clip": per_clip,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vowel-dir", type=Path, default=None)
    p.add_argument("--pataka-dir", type=Path, default=None)
    p.add_argument("--smile-dir", type=Path, default=None)
    p.add_argument("--label", choices=["PD", "HC"], default=None,
                   help="ground-truth label — for printing correct/incorrect only")
    args = p.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    weights = cfg["fusion"]["weights"]

    print("=" * 62)
    print("ParkScreen — Quick Multimodal Score (no LLM layer)")
    print("=" * 62)

    scores: dict[str, float | None] = {"phonation": None, "ddk": None, "facial": None}

    # phonation
    if args.vowel_dir and args.vowel_dir.exists():
        print(f"\n[phonation]  dir: {args.vowel_dir}")
        s, m = _score_phonation(args.vowel_dir)
        scores["phonation"] = s
        if s is not None:
            reps = "  ".join(f"[{v}]×{n}" for v, n in m["reps_per_vowel"].items())
            print(f"  vowels present: {reps}  (skipped {m['n_files_skipped']} files)")
            for sk in m["skipped"]:
                print(f"    ! skipped {sk['file']}: {sk['reason']}")
            f = m["features"]
            print(f"  weighted features: jitter_local={f['jitter_local']*100:.3f}%  "
                  f"shimmer_local={f['shimmer_local']*100:.2f}%  "
                  f"HNR={f['hnr_mean_db']:.1f} dB  "
                  f"F0_mean={f['f0_mean_hz']:.0f} Hz  "
                  f"F0_std={f['f0_std_hz']:.2f} Hz")
            print(f"  → PD probability: {s:.3f}")
        else:
            print(f"  → N/A ({m.get('reason')})")
    else:
        print("\n[phonation]  N/A (no --vowel-dir)")

    # ddk
    if args.pataka_dir and args.pataka_dir.exists():
        print(f"\n[ddk]        dir: {args.pataka_dir}")
        s, m = _score_ddk(args.pataka_dir)
        scores["ddk"] = s
        if s is not None:
            print(f"  aggregated over {m['n_files']} PATAKA file(s): {m['files']}")
            f = m["features"]
            print(f"  key features: ddk_rate={f['ddk_rate_hz']:.2f} syl/s  "
                  f"isi_cv={f['isi_cv']:.3f}  "
                  f"amp_cv={f['amp_cv']:.3f}  "
                  f"amp_decrement={f['amp_decrement']:.4f}")
            print(f"  → PD probability: {s:.3f}")
        else:
            print(f"  → N/A ({m.get('reason')})")
    else:
        print("\n[ddk]        N/A (no --pataka-dir)")

    # facial
    if args.smile_dir and args.smile_dir.exists():
        print(f"\n[facial]     dir: {args.smile_dir}")
        s, m = _score_facial(args.smile_dir)
        scores["facial"] = s
        for c in m.get("per_clip", []):
            sc = f"{c['score']:.3f}" if c["score"] is not None else "withheld"
            det = c["detection_rate"]
            det_s = f"{det:.2%}" if det is not None else "n/a"
            print(f"  {c['file']}: score={sc}  detection={det_s}")
            for w in c.get("warnings", []):
                print(f"    ! {w}")
        if s is not None:
            print(f"  → max PD probability over {m['n_clips_used']} clip(s): "
                  f"{s:.3f}  (selected: {m['selected_clip']})")
        else:
            print(f"  → N/A ({m.get('reason')})")
    else:
        print("\n[facial]     N/A (no --smile-dir)")

    # fusion
    print("\n" + "-" * 62)
    threshold = float(cfg["fusion"].get("agreement_threshold", 0.30))
    result = fuse_scores(scores, weights, agreement_threshold=threshold)
    fused = result["fused_score"]
    if fused is None:
        print("No channels scored — cannot fuse.")
        return

    wn = result["weights_normalized"]
    weight_str = "  ".join(f"{c}={w:.2f}" for c, w in wn.items())
    print(f"Fusion weights (renormalized over present channels): {weight_str}")
    print(f"FUSED PD probability: {fused:.3f}")
    pred = "PD" if fused >= 0.5 else "HC"
    print(f"Predicted class: {pred}  (threshold 0.5)")
    ag = result["agreement"]
    if ag["speech_channels_agree"] is not None:
        print(f"speech channels agree: {ag['speech_channels_agree']}")
    if ag["facial_agrees_with_speech"] is not None:
        print(f"facial agrees with speech: {ag['facial_agrees_with_speech']}")
    if ag["any_flag_for_review"]:
        print("⚠ per-channel disagreement — flag for clinical review")
    if args.label:
        correct = pred == args.label
        mark = "CORRECT" if correct else "WRONG"
        print(f"Ground truth: {args.label}  →  [{mark}]")


if __name__ == "__main__":
    main()
