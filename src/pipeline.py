"""ParkScreen end-to-end demo entry point: three task-matched uploads → Claude report.

Composes the three per-channel classifiers (phonation, DDK, facial) with the
score-level fusion and the Claude report layer. Same per-channel scoring
logic as ``src.fusion.quick_score`` — that module remains as the no-LLM
sanity check for debugging; this module adds the Claude context + report
step and uses a single OpenFace Docker run per smile clip
(``predict_and_summarize`` returns both the classifier score and the
hypomimia summary from the same CSV, matching the "one OpenFace run per
demo" invariant in CLAUDE.md Facial Features).

Any of the three upload dirs may be omitted → that channel returns N/A, the
fusion weights renormalize over the present channels, and Claude skips the
corresponding narrative section (both behaviours are enforced by
``fuse_scores`` and the frozen system prompt in ``claude_client.py``).

CLI:
    python -m src.pipeline \\
        --vowel-dir data/samples/pd_demo/vowel \\
        --pataka-dir data/samples/pd_demo/pataka \\
        --smile-dir data/samples/pd_demo/smile \\
        [--out-dir out/pd_demo] [--no-claude] [--label PD]

Gradio wrapper (Day 5 UI) is a thin shim over ``run_pipeline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.fusion.llm_fusion import build_claude_context, fuse_scores
from src.fusion.quick_score import _score_ddk, _score_phonation
from src.vision.predict_smile_pd import predict_and_summarize

REPO_ROOT = Path(__file__).resolve().parents[1]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _score_facial(smile_dir: Path) -> tuple[float | None, dict, dict | None]:
    """Score every smile clip in ``smile_dir``; return (score, meta, summary).

    One OpenFace Docker run per clip (``predict_and_summarize``). Max-pool
    over clips — pick the clip most suggestive of PD, and keep that clip's
    hypomimia summary as the one Claude sees. Matches
    ``quick_score._score_facial`` in fusion behaviour; adds the summary
    threading so the classifier score and the narrative come from the same
    frames.
    """
    per_clip: list[dict] = []
    # (score, name, summary) — only clips that passed the quality gate.
    scored_clips: list[tuple[float, str, dict | None]] = []

    for f in sorted(smile_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        r = predict_and_summarize(f)
        per_clip.append({
            "file": f.name,
            "score": r["score"],
            "detection_rate": r["detection_rate"],
            "warnings": r["warnings"],
        })
        if r["score"] is not None:
            scored_clips.append((r["score"], f.name, r.get("summary")))

    if not scored_clips:
        return None, {
            "reason": "all smile clips withheld by quality gate",
            "per_clip": per_clip,
        }, None

    best_score, best_file, best_summary = max(scored_clips, key=lambda t: t[0])
    return float(best_score), {
        "n_clips_used": len(scored_clips),
        "policy": "max",
        "selected_clip": best_file,
        "per_clip": per_clip,
    }, best_summary


def run_pipeline(
    vowel_dir: Path | str | None = None,
    pataka_dir: Path | str | None = None,
    smile_dir: Path | str | None = None,
    call_claude: bool = True,
) -> dict:
    """Score three task-matched upload dirs, fuse, and (optionally) generate a report.

    Args:
        vowel_dir: directory of sustained-vowel audio files. Filenames must
            follow ``<group>_<VOWEL><REP>.<ext>`` (e.g. ``PD_I1.m4a``) so the
            vowel-filter + weighted aggregation from training can be applied
            (Day 4 pipeline: [I, O, U] only). Missing dir → phonation N/A.
        pataka_dir: directory of PATAKA (/pa-ta-ka/) audio files. Missing dir
            → DDK N/A.
        smile_dir: directory of smile-task video files (mp4/mov/avi/mkv).
            Missing dir → facial N/A.
        call_claude: when False, skip the Anthropic SDK call and return the
            XML context string but no rendered report. Useful for debugging
            or running without an API key. When True and no channels have a
            fused score (all N/A), the Claude call is skipped anyway with a
            note in ``channel_meta``.

    Returns a dict:
        scores:          {channel: prob | None}
        channels:        {channel: {score, features} | None}     for Claude
        channel_meta:    per-channel diagnostics (files used, warnings, ...)
        facial_summary:  hypomimia JSON of the selected clip, or None
        fusion:          output of ``fuse_scores``
        claude_context:  XML block passed to Claude
        report:          markdown report (str), or None if the call was
                         skipped / failed
    """
    cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    weights = cfg["fusion"]["weights"]
    threshold = float(cfg["fusion"].get("agreement_threshold", 0.30))

    scores: dict[str, float | None] = {"phonation": None, "ddk": None, "facial": None}
    channels: dict[str, dict | None] = {"phonation": None, "ddk": None, "facial": None}
    channel_meta: dict[str, dict] = {}
    facial_summary: dict | None = None

    vowel_dir = Path(vowel_dir) if vowel_dir else None
    pataka_dir = Path(pataka_dir) if pataka_dir else None
    smile_dir = Path(smile_dir) if smile_dir else None

    # Phonation
    if vowel_dir is not None and vowel_dir.exists():
        score, meta = _score_phonation(vowel_dir)
        scores["phonation"] = score
        channel_meta["phonation"] = meta
        if score is not None:
            channels["phonation"] = {"score": score, "features": meta["features"]}

    # DDK
    if pataka_dir is not None and pataka_dir.exists():
        score, meta = _score_ddk(pataka_dir)
        scores["ddk"] = score
        channel_meta["ddk"] = meta
        if score is not None:
            channels["ddk"] = {"score": score, "features": meta["features"]}

    # Facial (single OpenFace Docker run per clip — score + summary from same CSV)
    if smile_dir is not None and smile_dir.exists():
        score, meta, summary = _score_facial(smile_dir)
        scores["facial"] = score
        channel_meta["facial"] = meta
        if score is not None:
            channels["facial"] = {"score": score}
            facial_summary = summary

    # Fusion — weights renormalize over present channels; N/A drops out
    fusion_result = fuse_scores(scores, weights, agreement_threshold=threshold)

    # Claude context (built even if we skip the API call — useful for debug)
    context_str = build_claude_context(channels, facial_summary, fusion_result)

    report_md: str | None = None
    if call_claude and fusion_result["fused_score"] is not None:
        # Lazy import: keeps `--no-claude` runs free of the anthropic SDK /
        # .env loading side effects.
        try:
            from src.report.claude_client import generate_report
            report_md = generate_report(context_str)
        except Exception as e:
            channel_meta["report_error"] = f"{type(e).__name__}: {e}"

    return {
        "scores": scores,
        "channels": channels,
        "channel_meta": channel_meta,
        "facial_summary": facial_summary,
        "fusion": fusion_result,
        "claude_context": context_str,
        "report": report_md,
    }


def _print_summary(result: dict, label: str | None = None) -> None:
    """Human-readable stdout dump for the CLI. Mirrors quick_score layout."""
    print("=" * 62)
    print("ParkScreen — Multimodal PD Screening Pipeline")
    print("=" * 62)

    scores = result["scores"]
    meta = result["channel_meta"]

    # Phonation
    if scores["phonation"] is not None:
        m = meta["phonation"]
        f = m["features"]
        reps = "  ".join(f"[{v}]×{n}" for v, n in m["reps_per_vowel"].items())
        print(f"\n[phonation]  vowels: {reps}  (skipped {m['n_files_skipped']})")
        print(f"  jitter_local={f['jitter_local']*100:.3f}%  "
              f"shimmer_local={f['shimmer_local']*100:.2f}%  "
              f"HNR={f['hnr_mean_db']:.1f} dB  "
              f"F0_mean={f['f0_mean_hz']:.0f} Hz")
        print(f"  → PD probability: {scores['phonation']:.3f}")
    elif "phonation" in meta:
        print(f"\n[phonation]  N/A ({meta['phonation'].get('reason')})")
    else:
        print("\n[phonation]  N/A (no --vowel-dir)")

    # DDK
    if scores["ddk"] is not None:
        m = meta["ddk"]
        f = m["features"]
        print(f"\n[ddk]        {m['n_files']} PATAKA file(s): {m['files']}")
        print(f"  ddk_rate={f['ddk_rate_hz']:.2f} syl/s  "
              f"isi_cv={f['isi_cv']:.3f}  "
              f"amp_cv={f['amp_cv']:.3f}  "
              f"amp_decrement={f['amp_decrement']:.4f}")
        print(f"  → PD probability: {scores['ddk']:.3f}")
    elif "ddk" in meta:
        print(f"\n[ddk]        N/A ({meta['ddk'].get('reason')})")
    else:
        print("\n[ddk]        N/A (no --pataka-dir)")

    # Facial
    if scores["facial"] is not None:
        m = meta["facial"]
        print(f"\n[facial]     {m['n_clips_used']} clip(s), selected: {m['selected_clip']}")
        for c in m.get("per_clip", []):
            sc = f"{c['score']:.3f}" if c["score"] is not None else "withheld"
            det = c["detection_rate"]
            det_s = f"{det:.2%}" if det is not None else "n/a"
            print(f"  {c['file']}: score={sc}  detection={det_s}")
            for w in c.get("warnings", []):
                print(f"    ! {w}")
        print(f"  → max PD probability: {scores['facial']:.3f}")
    elif "facial" in meta:
        m = meta["facial"]
        print(f"\n[facial]     N/A ({m.get('reason')})")
        for c in m.get("per_clip", []):
            det = c["detection_rate"]
            det_s = f"{det:.2%}" if det is not None else "n/a"
            print(f"  {c['file']}: score=withheld  detection={det_s}")
    else:
        print("\n[facial]     N/A (no --smile-dir)")

    # Fusion
    fusion = result["fusion"]
    print("\n" + "-" * 62)
    if fusion["fused_score"] is None:
        print("No channels scored — cannot fuse.")
        return

    wn = fusion["weights_normalized"]
    print("Fusion weights (renormalized): " + "  ".join(f"{c}={w:.2f}" for c, w in wn.items()))
    print(f"FUSED PD probability: {fusion['fused_score']:.3f}")
    pred = "PD" if fusion["fused_score"] >= 0.5 else "HC"
    print(f"Predicted class: {pred}  (threshold 0.5)")
    ag = fusion["agreement"]
    if ag["speech_channels_agree"] is not None:
        print(f"speech channels agree: {ag['speech_channels_agree']}")
    if ag["facial_agrees_with_speech"] is not None:
        print(f"facial agrees with speech: {ag['facial_agrees_with_speech']}")
    if ag["any_flag_for_review"]:
        print("⚠ per-channel disagreement — flag for clinical review")
    if label:
        correct = pred == label
        mark = "CORRECT" if correct else "WRONG"
        print(f"Ground truth: {label}  →  [{mark}]")

    # Claude report
    if result["report"]:
        print("\n" + "=" * 62)
        print("CLAUDE REPORT")
        print("=" * 62)
        print(result["report"])
    elif "report_error" in meta:
        print(f"\n(Claude call failed: {meta['report_error']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vowel-dir", type=Path, default=None,
                   help="Directory of vowel WAV/m4a files (filenames like *_I1.m4a).")
    p.add_argument("--pataka-dir", type=Path, default=None,
                   help="Directory of PATAKA WAV/m4a files.")
    p.add_argument("--smile-dir", type=Path, default=None,
                   help="Directory of smile-task video files (mp4/mov).")
    p.add_argument("--no-claude", action="store_true",
                   help="Skip the Claude API call — return context XML only.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If set, write context.xml + report.md + result.json here.")
    p.add_argument("--label", choices=["PD", "HC"], default=None,
                   help="Ground-truth label — prints correct/incorrect after fusion.")
    args = p.parse_args()

    result = run_pipeline(
        vowel_dir=args.vowel_dir,
        pataka_dir=args.pataka_dir,
        smile_dir=args.smile_dir,
        call_claude=not args.no_claude,
    )

    _print_summary(result, label=args.label)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "context.xml").write_text(result["claude_context"])
        if result["report"]:
            (args.out_dir / "report.md").write_text(result["report"])
        # Result JSON — strip DataFrames and non-serializable fields.
        serializable = {
            "scores": result["scores"],
            "channels": result["channels"],
            "channel_meta": result["channel_meta"],
            "facial_summary": result["facial_summary"],
            "fusion": result["fusion"],
        }
        (args.out_dir / "result.json").write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
