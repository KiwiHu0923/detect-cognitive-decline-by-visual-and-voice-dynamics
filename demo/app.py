"""ParkScreen — Gradio UI shell.

Three task-matched upload buckets (vowel / PATAKA / smile) → ``run_pipeline``
(the real Layer-2 orchestrator in ``src/pipeline.py``) → the same two-tier
report layout the mock UI already used (grade badge → Vocal + Facial modality
cards → per-channel accordions → agreement banner → Claude narrative).

Flow: upload view → click Analyze → progress bar → report view (with a
"New scan" button to return). Any bucket may be empty; that channel is
reported N/A and fusion renormalizes over the channels that were provided
(matches ``fuse_scores`` behaviour and the frozen system prompt in
``claude_client.py``).
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

import yaml

from demo.report_pdf import build_report_pdf
from src.fusion.llm_fusion import build_claude_context, fuse_scores
from src.fusion.quick_score import _score_ddk, _score_phonation
from src.pipeline import _score_facial
from src.report.claude_client import generate_report

REPO_ROOT = Path(__file__).resolve().parents[1]


VOWEL_TAB_INTRO = """**Sustained vowel recordings** — used to extract phonation features (jitter, shimmer, HNR, F0 stability) that reflect voice-quality perturbations characteristic of Parkinson's disease.

- Say each vowel **steadily** for **~3–5 seconds** per recording
- Use a **comfortable pitch and loudness** — no shouting, no whispering
- Try to hold pitch as **flat and stable** as possible; don't let it drift up or down
- Take a **full breath first**, then produce a single continuous sound (avoid pausing mid-vowel)
- Record in a **quiet room** with the mic close to your mouth (~10–15 cm)
- Upload **up to 3 recordings per vowel**; multiple reps are averaged for a more stable estimate
- **No filename convention needed** — the sub-section you upload to determines the vowel
- At least **one recording** from any of {/i/, /o/, /u/} is required to score this channel
"""

VOWEL_I_INSTRUCTIONS = "**Pronounce /i/**. Steady, sustained tone. Up to 3 reps."
VOWEL_O_INSTRUCTIONS = "**Pronounce /o/**. Steady, sustained tone. Up to 3 reps."
VOWEL_U_INSTRUCTIONS = "**Pronounce /u/**. Steady, sustained tone. Up to 3 reps."

PATAKA_INSTRUCTIONS = """**Rapid /pa-ta-ka/ repetition** — the canonical clinical DDK (diadochokinetic) task; measures articulatory speed and rhythmic regularity, key markers of the motor-speech signs in Parkinson's disease.

- Repeat "**pa-ta-ka**" **as fast and evenly as you can** for the full duration of the recording
- Each recording should be **~5 seconds** long
- Keep the **rhythm consistent** — try not to speed up, slow down, or pause between syllables
- Enunciate each of *pa* / *ta* / *ka* clearly — don't blur them together
- Stay at a **comfortable loudness**; don't shout, but also don't trail off toward the end
- Take a **full breath first** so you don't run out of air mid-recording
- Record in a **quiet room** with the mic close to your mouth (~10–15 cm)
- **1 or more recordings** accepted — multiple reps are averaged
- Any audio format (m4a, mp3, wav, flac, aac)
"""

SMILE_INSTRUCTIONS = """**Smile ×3 task** — used to extract facial hypomimia markers (reduced facial expressivity is a hallmark of PD). Follows the Islam et al. 2023 protocol used to train the underlying classifier.

**What to record:**
- **Smile 3 times** during the recording, each smile ~2–3 seconds long
- Between smiles, **relax to a neutral (resting) face** for ~1–2 seconds each time
- Pattern: neutral → smile #1 → neutral → smile #2 → neutral → smile #3 → neutral
- Total clip length: **8–12 seconds**

**How to smile (this matters a lot):**
- Smile as **dramatically as possible** — a big, wide, "posed for a photo" smile
- **Show your teeth**; engage the whole face (cheeks lifted, corners of mouth pulled up and outward)
- **Do NOT do a subtle, closed-lip grin** — the classifier is trained on exaggerated smile onsets
- Return to a genuinely **neutral, relaxed face** between smiles (no lingering half-smile)

**Framing and environment:**
- Face must be **clearly visible and well-lit** — position the camera roughly at **eye level**
- **Avoid backlight** (do not sit with a window behind you); light source should be in front
- Whole face should be in frame from **forehead to chin**, roughly centred
- Keep your **head roughly still** during the recording — no tilting or turning
- **Remove glasses** if possible (they can interfere with blink / AU detection)
- Remove hats or anything that shadows the eyes or mouth

**File requirements:**
- **Video format only** (mp4, mov, avi, mkv)
- Multiple clips accepted — the classifier score is **max-pooled** across clips, and the hypomimia summary is taken from the best-scoring clip
- Recommend recording **2–3 separate clips** and uploading all of them (increases robustness)
"""

# Example recordings shown under each task (see docs/DATASETS.md — self-recorded HC volunteer)
EXAMPLE_I_PATH = "data/samples/hc_demo/vowel/HC_I1.m4a"
EXAMPLE_O_PATH = "data/samples/hc_demo/vowel/HC_O1.m4a"
EXAMPLE_U_PATH = "data/samples/hc_demo/vowel/HC_U1.m4a"
EXAMPLE_PATAKA_PATH = "data/samples/hc_demo/pataka/HC_PATAKA1.m4a"
EXAMPLE_SMILE_PATH = "data/samples/hc_demo/smile/smile-instruction.gif"


OVERVIEW = """**What is ParkScreen?** ParkScreen combines three complementary signals to estimate a screening probability of Parkinson's disease from short at-home recordings — no clinician needed to record, no diagnostic claim made.

**Three tasks, three signals.** You provide (1) sustained vowels /i/, /o/, /u/ for **voice quality** (jitter, shimmer, HNR), (2) rapid /pa-ta-ka/ repetition for **articulatory speed and rhythm regularity**, and (3) a short smile-task video for **facial expressivity** (hypomimia). Each channel is scored by a small, interpretable classifier trained on published PD corpora — NeuroVoz for speech, UFNet for the smile task.

**How the fusion works.** The three per-channel probabilities are combined via AUC-excess weighting, then handed to Claude Opus 4.7 as structured evidence to synthesize into a clinical-style report. If a channel is missing or fails a quality gate, the fusion renormalizes over the remaining channels — nothing is silently reconciled.

**Validation.** Subject-level leave-one-subject-out on **49 PD × 46 age-matched controls**: speech-only fusion AUC **0.758**. Facial classifier in-distribution AUROC **0.812**.

**Important.** ParkScreen is a **screening decision-aid, not a diagnosis**. Every report carries the diagnostic disclaimer and a training-distribution caveat (all PD training subjects were medicated).

*Note: the demo runs on a serverless container that idles after ~15 minutes of inactivity. If you're the first visitor after an idle period, the initial page load may take a few seconds while the service wakes up — please be patient.*
"""


DISCLAIMER = (
    "<strong>Screening decision-aid, not a diagnosis.</strong> "
    "ParkScreen provides supporting evidence for clinical review only. "
    "It is not a substitute for evaluation by a qualified healthcare "
    "professional and must not be used for self-diagnosis."
)


def _processing_html(from_pct: int, to_pct: int, stage: str, duration_s: float = 0.4) -> str:
    """Render the processing-view HTML shown during analyze().

    The bar CSS-animates from ``from_pct`` to ``to_pct`` over ``duration_s``
    seconds — giving visible continuous motion between stage yields instead
    of discrete jumps. Gradio replaces the whole HTML fragment on each yield
    (so a CSS `transition` on the bar won't fire — the element is fresh each
    time); a per-yield `@keyframes` animation is the workaround. The keyframe
    name is unique per (from, to) so a stale definition can't be picked up.
    """
    from_pct = max(0, min(100, int(from_pct)))
    to_pct = max(0, min(100, int(to_pct)))
    anim = f"psfill_{from_pct}_{to_pct}"
    return f"""
<style>
@keyframes {anim} {{ from {{ width: {from_pct}%; }} to {{ width: {to_pct}%; }} }}
</style>
<div class="ps-processing__spinner">⚙️</div>
<div class="ps-processing__title">{stage}</div>
<div class="ps-progressbar">
  <div class="ps-progressbar__fill" style="width: {to_pct}%; animation: {anim} {duration_s}s linear forwards;"></div>
</div>
<div class="ps-processing__percent">{to_pct}%</div>
<div class="ps-processing__sub">Typically 30–90 seconds. Please don't close this tab.</div>
"""


CHANNEL_LABELS = {
    "phonation": "Phonation",
    "ddk": "Articulation (DDK)",
    "smile": "Smile task",
}


MODALITY_LABELS = {"vocal": "Vocal", "facial": "Facial"}


GRADE_LABELS = {
    "A": "Low likelihood of features consistent with PD",
    "B": "Moderate likelihood of features consistent with PD",
    "C": "Elevated likelihood of features consistent with PD",
}


# ---------------------------------------------------------------------------
# HTML rendering — small helpers that turn a report dict into styled markup.
# ---------------------------------------------------------------------------

def render_grade_badge(report: dict[str, Any]) -> str:
    grade = report["risk_grade"]
    fused = report["fused_score"]
    desc = GRADE_LABELS[grade]
    cls = f"ps-grade ps-grade--{grade.lower()}"
    return f"""
<div class="{cls}">
  <div class="ps-grade__letter">{grade}</div>
  <div class="ps-grade__meta">
    <div class="ps-grade__label">Overall risk grade</div>
    <div class="ps-grade__desc">{desc}</div>
    <div class="ps-grade__score">Fused score {fused:.2f}</div>
  </div>
</div>
"""


def _score_severity(score: float) -> str:
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "moderate"
    return "elevated"


def render_modality_card(key: str, mod: dict[str, Any]) -> str:
    name = MODALITY_LABELS[key]
    score = mod["score"]
    score_pct = int(round(score * 100))
    weight_pct = int(round(mod["weight"] * 100))
    severity = _score_severity(score)

    # SVG donut: arc length proportional to score. r=42 leaves room for a
    # 10-wide stroke inside the 100x100 viewBox.
    r = 42
    circ = 2 * math.pi * r
    arc = circ * max(0.0, min(1.0, score))
    gap = circ - arc

    donut = f"""
    <svg viewBox="0 0 100 100" class="ps-donut ps-donut--{severity}">
      <circle cx="50" cy="50" r="{r}" class="ps-donut__track"/>
      <circle cx="50" cy="50" r="{r}" class="ps-donut__arc"
              stroke-dasharray="{arc:.2f} {gap:.2f}"
              transform="rotate(-90 50 50)"/>
      <text x="50" y="47" text-anchor="middle" dominant-baseline="central" class="ps-donut__num">{score_pct}</text>
      <text x="50" y="65" text-anchor="middle" dominant-baseline="central" class="ps-donut__den">/ 100</text>
    </svg>
    """

    return f"""
<div class="ps-mod ps-mod--{severity}">
  <div class="ps-mod__header">
    <span class="ps-mod__title">{name}</span>
    <span class="ps-mod__weight">weight {weight_pct}%</span>
  </div>
  <div class="ps-mod__donutwrap">{donut}</div>
  <div class="ps-mod__summary">{mod['summary']}</div>
</div>
"""


def _channel_card(key: str, ch: dict[str, Any]) -> str:
    name = CHANNEL_LABELS[key]
    weight_pct = f"{int(round(ch['weight'] * 100))}%"
    if ch["status"] == "ok":
        score = ch["score"]
        bar_width = max(0.0, min(1.0, score)) * 100
        body = f"""
        <div class="ps-card__scorerow">
          <div class="ps-card__barouter"><div class="ps-card__barinner" style="width:{bar_width:.1f}%"></div></div>
          <div class="ps-card__scorenum">{score:.2f}</div>
        </div>
        <div class="ps-card__meta">Confidence {ch['confidence']:.2f}</div>
        <div class="ps-card__note">{ch['note']}</div>
        """
        status_class = "ps-card--ok"
    else:
        body = f"""
        <div class="ps-card__na">N/A — task not detected</div>
        <div class="ps-card__note">{ch['note']}</div>
        """
        status_class = "ps-card--na"
    return f"""
<div class="ps-card {status_class}">
  <div class="ps-card__header">
    <span class="ps-card__title">{name}</span>
    <span class="ps-card__weight">{weight_pct} of channel</span>
  </div>
  <div class="ps-card__body">{body}</div>
</div>
"""


def render_channels_of(mod: dict[str, Any]) -> str:
    channels = mod["channels"]
    cards = "\n".join(_channel_card(k, v) for k, v in channels.items())
    return f'<div class="ps-grid">{cards}</div>'


def render_agreement(report: dict[str, Any]) -> str:
    agreement = report["agreement"]
    copy = {
        "agree": ("Channels agree", "All active channels point in the same direction. Higher confidence in the fused score."),
        "partial": ("Channels partially agree", "Speech channels align; facial signal is weaker but directionally consistent."),
        "disagree": ("Channels disagree", "Active channels point in different directions. Flag for clinical review."),
    }[agreement]
    return f"""
<div class="ps-agreement ps-agreement--{agreement}">
  <div class="ps-agreement__title">{copy[0]}</div>
  <div class="ps-agreement__body">{copy[1]}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Pipeline result → UI report dict mapping
# ---------------------------------------------------------------------------

def _grade_from_score(fused: float) -> str:
    """Bucket the fused probability into an A/B/C grade.

    Thresholds mirror ``_score_severity`` (low/moderate/elevated) so the grade
    letter and the modality-card donut colour land in the same bin.
    """
    if fused < 0.4:
        return "A"
    if fused < 0.7:
        return "B"
    return "C"


def _agreement_label(agreement: dict[str, bool | None]) -> str:
    """Map ``fuse_scores`` agreement dict → three-state UI label.

    - both flags True (or one True + one undefined) → agree
    - both flags False → disagree
    - anything else (one True + one False, or both undefined) → partial
    """
    speech = agreement["speech_channels_agree"]
    facial = agreement["facial_agrees_with_speech"]
    votes = [x for x in (speech, facial) if x is not None]
    if not votes:
        return "partial"
    if all(votes):
        return "agree"
    if not any(votes):
        return "disagree"
    return "partial"


def _phonation_note(features: dict[str, float] | None) -> str:
    """Short auto-note from the interpretable phonation features."""
    if not features:
        return "No phonation features extracted."
    j = features.get("jitter_local", 0.0) * 100.0        # ratio → percent
    h = features.get("hnr_mean_db", 0.0)
    fs = features.get("f0_std_hz", 0.0)
    return f"jitter {j:.2f}%, HNR {h:.1f} dB, F0 SD {fs:.1f} Hz"


def _ddk_note(features: dict[str, float] | None) -> str:
    if not features:
        return "No DDK features extracted."
    rate = features.get("ddk_rate_hz", 0.0)
    isi_cv = features.get("isi_cv", 0.0)
    amp_cv = features.get("amp_cv", 0.0)
    return f"rate {rate:.2f} syl/s, ISI CV {isi_cv:.2f}, amp CV {amp_cv:.2f}"


def _smile_note(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "No hypomimia summary."
    amp = summary.get("AU12_amplitude_on_smile_cue")
    blink = summary.get("blink_rate_per_min")
    parts = []
    if amp is not None:
        parts.append(f"AU12 peak {amp:.2f}")
    if blink is not None:
        parts.append(f"blink {blink:.1f}/min")
    return ", ".join(parts) if parts else "hypomimia markers unavailable"


def _channel_confidence(channel_meta: dict[str, Any] | None, channel: str) -> float:
    """Cheap confidence proxy for the UI card — quantity of input, not model uncertainty."""
    if channel_meta is None:
        return 0.0
    if channel == "phonation":
        n = int(channel_meta.get("n_files_used", 0))
        return min(1.0, n / 3.0)
    if channel == "ddk":
        n = int(channel_meta.get("n_files", 0))
        return min(1.0, n / 3.0)
    if channel == "facial":
        # Use detection rate of the selected clip as a proxy.
        selected = channel_meta.get("selected_clip")
        for c in channel_meta.get("per_clip", []):
            if c["file"] == selected:
                return float(c["detection_rate"] or 0.0)
        return 0.0
    return 0.0


def _map_pipeline_to_report(result: dict) -> dict[str, Any]:
    """Turn ``run_pipeline`` output into the report dict the UI renders.

    Fields map:
      - ``risk_grade`` / ``fused_score`` come from ``result['fusion']``.
      - Modality "vocal" wraps phonation + DDK; its score is the fusion-weight-
        weighted average of the two present speech channels (or the single one
        present); its channel weights are renormalized within Vocal.
      - Modality "facial" wraps the smile channel only.
      - ``agreement`` is derived from ``fuse_scores`` flags.
      - ``narrative`` is the Claude markdown (or a placeholder if we skipped
        the Claude call — should not happen on the UI path).
    """
    fusion = result["fusion"]
    channels_in = result["channels"]
    meta = result["channel_meta"]
    w_norm = fusion.get("weights_normalized", {})
    facial_summary = result.get("facial_summary")

    fused = fusion.get("fused_score")
    if fused is None:
        # No channels scored — nothing to render. Caller validates before us,
        # but keep a safe fallback so the UI doesn't KeyError.
        return {
            "risk_grade": "A",
            "fused_score": 0.0,
            "modalities": {"vocal": None, "facial": None},
            "agreement": "partial",
            "narrative": "No channels could be scored. Check the recording protocol and retry.",
        }

    # ---- Vocal modality (phonation + DDK) ----
    phon = channels_in.get("phonation")
    ddk = channels_in.get("ddk")
    phon_w_global = w_norm.get("phonation", 0.0)
    ddk_w_global = w_norm.get("ddk", 0.0)
    vocal_weight = phon_w_global + ddk_w_global

    vocal_channels: dict[str, Any] = {}
    if phon is not None:
        vocal_channels["phonation"] = {
            "status": "ok",
            "score": phon["score"],
            "confidence": _channel_confidence(meta.get("phonation"), "phonation"),
            "weight": phon_w_global / vocal_weight if vocal_weight else 0.0,
            "note": _phonation_note(phon.get("features")),
        }
    else:
        vocal_channels["phonation"] = {
            "status": "na_off_task",
            "score": 0.0,
            "confidence": 0.0,
            "weight": 0.0,
            "note": (meta.get("phonation", {}) or {}).get("reason") or "No vowel recordings provided.",
        }
    if ddk is not None:
        vocal_channels["ddk"] = {
            "status": "ok",
            "score": ddk["score"],
            "confidence": _channel_confidence(meta.get("ddk"), "ddk"),
            "weight": ddk_w_global / vocal_weight if vocal_weight else 0.0,
            "note": _ddk_note(ddk.get("features")),
        }
    else:
        vocal_channels["ddk"] = {
            "status": "na_off_task",
            "score": 0.0,
            "confidence": 0.0,
            "weight": 0.0,
            "note": (meta.get("ddk", {}) or {}).get("reason") or "No PATAKA recordings provided.",
        }

    if vocal_weight > 0:
        vocal_score = 0.0
        if phon is not None:
            vocal_score += (phon_w_global / vocal_weight) * phon["score"]
        if ddk is not None:
            vocal_score += (ddk_w_global / vocal_weight) * ddk["score"]
        vocal_summary_parts = []
        if phon is not None:
            vocal_summary_parts.append(f"phonation {phon['score']:.2f}")
        if ddk is not None:
            vocal_summary_parts.append(f"DDK {ddk['score']:.2f}")
        vocal_summary = "Speech channels: " + ", ".join(vocal_summary_parts) + "."
    else:
        vocal_score = 0.0
        vocal_summary = "No speech recordings provided — Vocal channel is N/A."

    vocal_mod = {
        "score": vocal_score,
        "weight": vocal_weight,
        "summary": vocal_summary,
        "channels": vocal_channels,
    }

    # ---- Facial modality (smile only) ----
    facial = channels_in.get("facial")
    facial_w_global = w_norm.get("facial", 0.0)
    facial_channels: dict[str, Any] = {}
    if facial is not None:
        facial_channels["smile"] = {
            "status": "ok",
            "score": facial["score"],
            "confidence": _channel_confidence(meta.get("facial"), "facial"),
            "weight": 1.0,
            "note": _smile_note(facial_summary),
        }
        facial_summary_str = f"Smile classifier {facial['score']:.2f}. " + _smile_note(facial_summary) + "."
    else:
        facial_channels["smile"] = {
            "status": "na_off_task",
            "score": 0.0,
            "confidence": 0.0,
            "weight": 0.0,
            "note": (meta.get("facial", {}) or {}).get("reason") or "No smile video provided.",
        }
        facial_summary_str = "No smile video provided — Facial channel is N/A."

    facial_mod = {
        "score": facial["score"] if facial is not None else 0.0,
        "weight": facial_w_global,
        "summary": facial_summary_str,
        "channels": facial_channels,
    }

    narrative = result.get("report") or (
        "_(Claude report unavailable — using structured signals only. See "
        "`channel_meta.report_error` in the CLI output for details.)_"
    )

    return {
        "risk_grade": _grade_from_score(fused),
        "fused_score": float(fused),
        "modalities": {"vocal": vocal_mod, "facial": facial_mod},
        "agreement": _agreement_label(fusion["agreement"]),
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def _stage_uploads(
    files: list | None,
    dest: Path,
    rename_prefix: str | None = None,
) -> int:
    """Copy uploaded files into ``dest``.

    If ``rename_prefix`` is given (e.g. ``"X_I"``), files are renamed to
    ``X_I1.ext``, ``X_I2.ext``, ... in upload order. This lets the vowel
    upload widgets skip the ``<group>_<VOWEL><REP>.<ext>`` filename
    convention entirely — the widget itself signals which vowel it is, and
    we forge a filename the pipeline's vowel parser can read. Without a
    ``rename_prefix`` the original filename is preserved (used for PATAKA
    and smile, where filename is not parsed).

    Returns the number of files copied.
    """
    if not files:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        # gr.File items are FileData (attribute .name) or a bare string path
        # depending on gradio version. Normalize.
        src = getattr(f, "name", None) or (f if isinstance(f, str) else None)
        if src is None:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        if rename_prefix:
            n += 1
            out_name = f"{rename_prefix}{n}{src_path.suffix}"
            shutil.copy2(src_path, dest / out_name)
        else:
            shutil.copy2(src_path, dest / src_path.name)
            n += 1
    return n


def analyze(
    vowel_i_files: list | None,
    vowel_o_files: list | None,
    vowel_u_files: list | None,
    pataka_files: list | None,
    smile_files: list | None,
):
    """Staged generator that runs the pipeline in-line so each channel's
    completion can update the processing view's progress bar.

    Yields tuples of 12 UI updates matching the click handler's ``outputs``
    list: ``[upload_view, processing_view, report_view, processing_html,
    grade_html, vocal_card_html, vocal_detail_html, facial_card_html,
    facial_detail_html, agreement_html, narrative_md, current_report]``.

    On any exception this yields back to the upload view before raising
    ``gr.Error`` so the user can fix the underlying issue and retry.
    """
    if not any([vowel_i_files, vowel_o_files, vowel_u_files, pataka_files, smile_files]):
        raise gr.Error("Upload at least one recording (vowels, PATAKA, or smile) before analyzing.")

    keep = gr.update()  # sentinel — leave the target component untouched

    # Tracks the last progress % emitted so the next yield's CSS keyframe
    # animation knows where to start from. Wrapped in a list so the nested
    # `stage_yield` closure can mutate it without a nonlocal declaration.
    prev_pct = [0]

    def stage_yield(to_pct: int, stage: str, duration_s: float = 0.5) -> tuple:
        """Yield an update that only refreshes the processing_html panel.

        ``duration_s`` is the estimated time for this stage — the bar
        CSS-animates from the previous % to ``to_pct`` over that many seconds,
        so long stages (facial ~20s, Claude ~15s) show continuous motion
        instead of freezing at one number.
        """
        from_pct = prev_pct[0]
        prev_pct[0] = to_pct
        return (
            keep,  # upload_view (unchanged from the initial visibility flip)
            keep,  # processing_view
            keep,  # report_view
            gr.update(value=_processing_html(from_pct, to_pct, stage, duration_s)),
            keep, keep, keep, keep, keep, keep, keep, keep,  # 8 report components
            keep,  # download_btn
        )

    # Yield #1 — switch to processing view + kick off progress bar at 5%.
    prev_pct[0] = 5
    yield (
        gr.update(visible=False),  # upload_view
        gr.update(visible=True),   # processing_view
        gr.update(visible=False),  # report_view
        gr.update(value=_processing_html(0, 5, "Staging uploads…", 1.0)),
        keep, keep, keep, keep, keep, keep, keep, keep,
        keep,  # download_btn
    )

    try:
        # ---------- Stage uploads --------------------------------------
        workdir = Path(tempfile.mkdtemp(prefix="parkscreen_ui_"))
        vowel_dir = workdir / "vowel"
        pataka_dir = workdir / "pataka"
        smile_dir = workdir / "smile"

        n_i = _stage_uploads(vowel_i_files, vowel_dir, rename_prefix="X_I")
        n_o = _stage_uploads(vowel_o_files, vowel_dir, rename_prefix="X_O")
        n_u = _stage_uploads(vowel_u_files, vowel_dir, rename_prefix="X_U")
        n_vowel = n_i + n_o + n_u
        n_pataka = _stage_uploads(pataka_files, pataka_dir)
        n_smile = _stage_uploads(smile_files, smile_dir)

        # Load fusion config once — mirrors src.pipeline.run_pipeline.
        cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
        weights = cfg["fusion"]["weights"]
        threshold = float(cfg["fusion"].get("agreement_threshold", 0.30))

        scores: dict[str, float | None] = {"phonation": None, "ddk": None, "facial": None}
        channels: dict[str, dict | None] = {"phonation": None, "ddk": None, "facial": None}
        channel_meta: dict[str, dict] = {}
        facial_summary: dict | None = None

        # ---------- Phonation ------------------------------------------
        yield stage_yield(28, "Extracting phonation features…", 5.0)
        if n_vowel:
            phon_score, phon_meta = _score_phonation(vowel_dir)
            scores["phonation"] = phon_score
            channel_meta["phonation"] = phon_meta
            if phon_score is not None:
                channels["phonation"] = {"score": phon_score, "features": phon_meta["features"]}

        # ---------- DDK ------------------------------------------------
        yield stage_yield(48, "Extracting DDK / articulation features…", 5.0)
        if n_pataka:
            ddk_score, ddk_meta = _score_ddk(pataka_dir)
            scores["ddk"] = ddk_score
            channel_meta["ddk"] = ddk_meta
            if ddk_score is not None:
                channels["ddk"] = {"score": ddk_score, "features": ddk_meta["features"]}

        # ---------- Facial (long — OpenFace) ---------------------------
        yield stage_yield(78, "Extracting facial features (OpenFace — this can take a moment)…", 20.0)
        if n_smile:
            facial_score, facial_meta, summary = _score_facial(smile_dir)
            scores["facial"] = facial_score
            channel_meta["facial"] = facial_meta
            if facial_score is not None:
                channels["facial"] = {"score": facial_score}
                facial_summary = summary

        # ---------- Fusion ---------------------------------------------
        yield stage_yield(85, "Fusing per-channel scores…", 1.0)
        fusion_result = fuse_scores(scores, weights, agreement_threshold=threshold)
        context_str = build_claude_context(channels, facial_summary, fusion_result)

        # ---------- Claude report --------------------------------------
        yield stage_yield(95, "Generating clinical report with Claude…", 15.0)
        report_md: str | None = None
        if fusion_result["fused_score"] is not None:
            try:
                report_md = generate_report(context_str)
            except Exception as e:
                channel_meta["report_error"] = f"{type(e).__name__}: {e}"

        # ---------- Render ---------------------------------------------
        yield stage_yield(100, "Rendering report…", 1.0)
        pipeline_result = {
            "scores": scores,
            "channels": channels,
            "channel_meta": channel_meta,
            "facial_summary": facial_summary,
            "fusion": fusion_result,
            "claude_context": context_str,
            "report": report_md,
        }
        report = _map_pipeline_to_report(pipeline_result)

    except Exception as exc:
        # Yield back to the upload view first, then raise gr.Error so the
        # toast fires and the UI stays in a retryable state.
        yield (
            gr.update(visible=True),   # upload_view
            gr.update(visible=False),  # processing_view
            gr.update(visible=False),  # report_view
            keep,
            keep, keep, keep, keep, keep, keep, keep, keep,
            keep,  # download_btn
        )
        exc_type = type(exc).__name__
        exc_str = str(exc)
        looks_like_docker = (
            "docker" in exc_str.lower()
            or "openface" in exc_str.lower()
            or exc_type == "CalledProcessError"
        )
        if looks_like_docker:
            msg = (
                "Facial extraction failed — most likely Docker Desktop is not running. "
                "Open Docker Desktop, wait until the whale icon in the menu bar is steady, "
                f"then click Analyze again. ({exc_type}: {exc_str[:200]})"
            )
        else:
            msg = f"Analysis failed: {exc_type}: {exc_str[:400]} (see terminal for the full traceback)."
        raise gr.Error(msg)

    # Pre-generate the PDF before showing the report so the DownloadButton's
    # value is set to a real file path the moment the user sees the button.
    # Otherwise DownloadButton needs two clicks: one to run the fn (which sets
    # value), one to actually download. Pre-setting the value → one click.
    pdf_dir = tempfile.mkdtemp(prefix="parkscreen_pdf_")
    pdf_path = os.path.join(pdf_dir, "parkscreen_report.pdf")
    try:
        with open(pdf_path, "wb") as f:
            build_report_pdf(report, f)
    except Exception:
        pdf_path = None  # PDF failure shouldn't block the report display

    # Yield #final — success: switch to report view with all content populated.
    yield (
        gr.update(visible=False),  # upload_view
        gr.update(visible=False),  # processing_view
        gr.update(visible=True),   # report_view
        keep,                      # processing_html (unchanged)
        render_grade_badge(report),
        render_modality_card("vocal", report["modalities"]["vocal"]),
        render_channels_of(report["modalities"]["vocal"]),
        render_modality_card("facial", report["modalities"]["facial"]),
        render_channels_of(report["modalities"]["facial"]),
        render_agreement(report),
        report["narrative"],
        report,
        gr.update(value=pdf_path),  # download_btn — pre-set so single click downloads
    )


def new_scan():
    """Return the UI to the upload view, clearing any prior report state."""
    return (
        gr.update(visible=True),   # upload_view
        gr.update(visible=False),  # processing_view (defensive; usually already hidden)
        gr.update(visible=False),  # report_view
        None,                      # vowel_i_input (clear)
        None,                      # vowel_o_input (clear)
        None,                      # vowel_u_input (clear)
        None,                      # pataka_input (clear)
        None,                      # smile_input (clear)
        None,                      # current_report state (clear)
        gr.update(value=None),     # download_btn — clear stale PDF path
    )


def prepare_pdf_download(current_report: dict[str, Any] | None) -> str:
    """Render the current session's report to a PDF and return the path.

    The button is only reachable from the report view (visible after Analyze
    populates ``current_report``), so a null state here means the UI wiring
    is wrong — surface it loudly instead of exporting a placeholder.
    """
    if current_report is None:
        raise gr.Error("No report to export — analyze a recording first.")
    tmpdir = tempfile.mkdtemp(prefix="parkscreen_")
    path = os.path.join(tmpdir, "parkscreen_report.pdf")
    with open(path, "wb") as f:
        build_report_pdf(current_report, f)
    return path


# ---------------------------------------------------------------------------
# Styling — dark clinical palette. Kept inline so the shell is one self-
# contained file for iteration.
# ---------------------------------------------------------------------------

CSS = """
/* Reserve scrollbar space so switching between tall/short tabs doesn't
   shift the horizontally-centred container by the scrollbar width. This
   is the actual cause of the "frame jumps sideways" symptom on tab switch. */
html {
  scrollbar-gutter: stable;
}

:root {
  --ps-surface: #131c2e;
  --ps-surface-2: #1c2740;
  --ps-border: #2a3552;
  --ps-border-strong: #384565;
  --ps-text: #e6edf7;
  --ps-text-muted: #a0aec8;
  --ps-text-faint: #6b7896;
  --ps-accent: #60a5fa;
  --ps-accent-soft: #93c5fd;
  --ps-risk-low: #10b981;
  --ps-risk-moderate: #f59e0b;
  --ps-risk-elevated: #ef4444;
}

.gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
  color: var(--ps-text);
  /* width: 100% forces the container to always claim the full 1100px cap;
     without it the container shrinks to fit the narrowest tab's natural
     width, which makes tabs appear at different widths on switch. */
  width: 100% !important;
  max-width: 1100px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Header */
.ps-header {
  padding: 12px 4px 4px 4px;
}
.ps-header__title {
  font-size: 26px;
  font-weight: 600;
  color: var(--ps-text);
  letter-spacing: -0.01em;
}
.ps-header__subtitle {
  font-size: 14px;
  color: var(--ps-text-muted);
  margin-top: 4px;
}
.ps-header__subtitle em {
  color: var(--ps-accent-soft);
  font-style: italic;
}

/* Grade badge — big letter + meta on the right */
.ps-grade {
  display: flex;
  align-items: center;
  gap: 22px;
  border-radius: 12px;
  padding: 22px 24px;
  border: 1px solid var(--ps-border);
  border-left-width: 4px;
  background: var(--ps-surface);
  margin-bottom: 20px;
}
.ps-grade--a { border-left-color: var(--ps-risk-low);       background: linear-gradient(90deg, rgba(16,185,129,0.10), var(--ps-surface) 60%); }
.ps-grade--b { border-left-color: var(--ps-risk-moderate);  background: linear-gradient(90deg, rgba(245,158,11,0.10), var(--ps-surface) 60%); }
.ps-grade--c { border-left-color: var(--ps-risk-elevated);  background: linear-gradient(90deg, rgba(239,68,68,0.12),  var(--ps-surface) 60%); }
.ps-grade__letter {
  font-size: 72px;
  font-weight: 700;
  line-height: 1;
  min-width: 100px;
  text-align: center;
  padding: 6px 10px;
  border-radius: 12px;
  background: var(--ps-surface-2);
}
.ps-grade--a .ps-grade__letter { color: var(--ps-risk-low); }
.ps-grade--b .ps-grade__letter { color: var(--ps-risk-moderate); }
.ps-grade--c .ps-grade__letter { color: var(--ps-risk-elevated); }
.ps-grade__label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ps-text-faint);
}
.ps-grade__desc {
  font-size: 18px;
  font-weight: 500;
  color: var(--ps-text);
  margin-top: 4px;
  line-height: 1.3;
}
.ps-grade__score {
  font-size: 13px;
  color: var(--ps-text-muted);
  margin-top: 6px;
}

/* Modality card — donut score for Vocal / Facial */
.ps-mod {
  background: var(--ps-surface);
  border: 1px solid var(--ps-border);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 4px;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.ps-mod__header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}
.ps-mod__title {
  font-size: 16px;
  font-weight: 600;
  color: var(--ps-text);
}
.ps-mod__weight {
  font-size: 11px;
  color: var(--ps-text-faint);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.ps-mod__donutwrap {
  display: flex;
  justify-content: center;
  padding: 8px 0 12px 0;
}
.ps-mod__summary {
  font-size: 13px;
  color: var(--ps-text-muted);
  line-height: 1.5;
  text-align: center;
}

/* Donut chart (SVG) */
.ps-donut {
  width: 150px;
  height: 150px;
  display: block;
}
.ps-donut__track {
  fill: none;
  stroke: var(--ps-surface-2);
  stroke-width: 10;
}
.ps-donut__arc {
  fill: none;
  stroke-width: 10;
  stroke-linecap: round;
  transition: stroke-dasharray 0.4s ease;
}
.ps-donut--low       .ps-donut__arc { stroke: var(--ps-risk-low); }
.ps-donut--moderate  .ps-donut__arc { stroke: var(--ps-risk-moderate); }
.ps-donut--elevated  .ps-donut__arc { stroke: var(--ps-risk-elevated); }
.ps-donut__num {
  fill: var(--ps-text);
  font-size: 22px;
  font-weight: 700;
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
}
.ps-donut__den {
  fill: var(--ps-text-faint);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.08em;
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
}

/* Channel stack inside a modality accordion — single column now */
.ps-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 6px 0 4px 0;
}
.ps-card {
  background: var(--ps-surface-2);
  border: 1px solid var(--ps-border);
  border-radius: 8px;
  padding: 12px 14px;
}
.ps-card--na {
  background: transparent;
  border-style: dashed;
  border-color: var(--ps-border);
}
.ps-card__header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.ps-card__title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ps-text);
}
.ps-card__weight {
  font-size: 10px;
  color: var(--ps-text-faint);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.ps-card__scorerow {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.ps-card__barouter {
  flex: 1;
  height: 6px;
  background: var(--ps-border);
  border-radius: 3px;
  overflow: hidden;
}
.ps-card__barinner {
  height: 100%;
  background: var(--ps-accent);
  border-radius: 3px;
}
.ps-card__scorenum {
  font-size: 14px;
  font-weight: 600;
  color: var(--ps-text);
  min-width: 40px;
  text-align: right;
}
.ps-card__meta {
  font-size: 12px;
  color: var(--ps-text-muted);
  margin-bottom: 4px;
}
.ps-card__note {
  font-size: 13px;
  color: var(--ps-text-muted);
  line-height: 1.4;
}
.ps-card__na {
  font-size: 13px;
  color: var(--ps-text-faint);
  font-weight: 500;
  margin-bottom: 6px;
}

/* Agreement banner */
.ps-agreement {
  border-left: 3px solid var(--ps-accent);
  background: var(--ps-surface);
  padding: 12px 16px;
  border-radius: 4px;
  margin: 20px 0 16px 0;
}
.ps-agreement--agree    { border-left-color: var(--ps-risk-low); }
.ps-agreement--partial  { border-left-color: var(--ps-risk-moderate); }
.ps-agreement--disagree { border-left-color: var(--ps-risk-elevated); }
.ps-agreement__title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ps-text);
}
.ps-agreement__body {
  font-size: 13px;
  color: var(--ps-text-muted);
  margin-top: 3px;
  line-height: 1.4;
}

/* Disclaimer */
.ps-disclaimer {
  font-size: 12px;
  color: var(--ps-text-muted);
  background: var(--ps-surface);
  border: 1px solid var(--ps-border);
  border-radius: 6px;
  padding: 12px 14px;
  margin-top: 20px;
  line-height: 1.5;
}

/* Fixed-height upload view — locks the Analyze button's vertical position
   regardless of which tab is active. This is the load-bearing rule; the
   per-panel min-height below is a backup for older Gradio DOM. */
.ps-upload-view {
  min-height: 720px !important;
  width: 100% !important;
}

/* High-level project overview shown at the top of the upload view */
.ps-overview {
  font-size: 13px;
  color: var(--ps-text-muted);
  line-height: 1.6;
  padding: 8px 4px 20px 4px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--ps-border);
}
.ps-overview p {
  margin: 0 0 10px 0;
}
.ps-overview strong {
  color: var(--ps-text);
  font-weight: 600;
}
.ps-tab-panel {
  min-height: 520px !important;
  width: 100% !important;
}

/* Smile-task example GIF — constrain the rendered image size WITHOUT
   changing the outer container width (so tab-container stays same width
   regardless of GIF file dimensions). The parent Column is unchanged. */
.ps-smile-example {
  max-width: 480px !important;
  margin: 0 auto !important;
}
.ps-smile-example img {
  max-height: 320px !important;
  width: auto !important;
  margin: 0 auto !important;
  display: block !important;
}

/* Processing view (analyze in progress) */
.ps-processing {
  padding: 60px 20px 40px;
  text-align: center;
  min-height: 400px;
}
@keyframes ps-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.ps-processing__spinner {
  font-size: 56px;
  display: inline-block;
  animation: ps-spin 1.6s linear infinite;
}
.ps-processing__title {
  margin-top: 24px;
  font-size: 18px;
  font-weight: 600;
  color: var(--ps-text);
}
.ps-progressbar {
  margin: 24px auto 8px auto;
  max-width: 420px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--ps-border);
  height: 10px;
  border-radius: 5px;
  overflow: hidden;
}
.ps-progressbar__fill {
  background: var(--ps-accent);
  height: 100%;
  border-radius: 5px;
  transition: width 0.4s ease;
}
.ps-processing__percent {
  font-size: 13px;
  color: var(--ps-text-muted);
  font-variant-numeric: tabular-nums;
}
.ps-processing__sub {
  margin-top: 24px;
  font-size: 13px;
  color: var(--ps-text-faint);
}
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="ParkScreen") as demo:
    gr.HTML(
        """
        <div class="ps-header">
          <div class="ps-header__title">ParkScreen</div>
          <div class="ps-header__subtitle">
            Multimodal screening decision-aid for Parkinson's disease signs — <em>not a diagnosis</em>.
          </div>
        </div>
        """
    )

    # =====================================================================
    # Upload view — three task-matched buckets (see Demo Protocol in CLAUDE.md)
    # =====================================================================
    with gr.Column(visible=True, elem_classes="ps-upload-view") as upload_view:
        gr.Markdown(OVERVIEW, elem_classes="ps-overview")
        with gr.Tabs():
            # --- Tab 1: Sustained vowels ------------------------------
            with gr.Tab("1 · Sustained vowels"):
                with gr.Column(elem_classes="ps-tab-panel"):
                    gr.Markdown(VOWEL_TAB_INTRO)
                    with gr.Row(equal_height=False):
                        with gr.Column():
                            gr.Markdown(VOWEL_I_INSTRUCTIONS)
                            vowel_i_input = gr.File(
                                label="Recordings of /i/ (up to 3)",
                                file_count="multiple",
                                file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
                                height=140,
                            )
                            gr.Audio(
                                value=EXAMPLE_I_PATH,
                                label="Example (HC volunteer, /i/)",
                                interactive=False,
                            )
                        with gr.Column():
                            gr.Markdown(VOWEL_O_INSTRUCTIONS)
                            vowel_o_input = gr.File(
                                label="Recordings of /o/ (up to 3)",
                                file_count="multiple",
                                file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
                                height=140,
                            )
                            gr.Audio(
                                value=EXAMPLE_O_PATH,
                                label="Example (HC volunteer, /o/)",
                                interactive=False,
                            )
                        with gr.Column():
                            gr.Markdown(VOWEL_U_INSTRUCTIONS)
                            vowel_u_input = gr.File(
                                label="Recordings of /u/ (up to 3)",
                                file_count="multiple",
                                file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
                                height=140,
                            )
                            gr.Audio(
                                value=EXAMPLE_U_PATH,
                                label="Example (HC volunteer, /u/)",
                                interactive=False,
                            )

            # --- Tab 2: PATAKA ----------------------------------------
            with gr.Tab("2 · Rapid /pa-ta-ka/"):
                with gr.Column(elem_classes="ps-tab-panel"):
                    gr.Markdown(PATAKA_INSTRUCTIONS)
                    pataka_input = gr.File(
                        label="PATAKA recordings (audio)",
                        file_count="multiple",
                        file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
                        height=140,
                    )
                    gr.Audio(
                        value=EXAMPLE_PATAKA_PATH,
                        label="Example (HC volunteer, one PATAKA rep)",
                        interactive=False,
                    )

            # --- Tab 3: Smile task ------------------------------------
            with gr.Tab("3 · Smile task"):
                with gr.Column(elem_classes="ps-tab-panel"):
                    gr.Markdown(SMILE_INSTRUCTIONS)
                    smile_input = gr.File(
                        label="Smile videos",
                        file_count="multiple",
                        file_types=["video", ".mp4", ".mov", ".avi", ".mkv"],
                        height=140,
                    )
                    gr.Image(
                        value=EXAMPLE_SMILE_PATH,
                        label="Example (HC volunteer, smile animation)",
                        interactive=False,
                        elem_classes="ps-smile-example",
                    )

        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    # =====================================================================
    # Processing view — shown while analyze() is running. The HTML gets
    # updated on each yield in analyze() to reflect current stage + %.
    # =====================================================================
    with gr.Column(visible=False, elem_classes="ps-processing") as processing_view:
        processing_html = gr.HTML(_processing_html(0, 0, "Preparing…", 0.4))

    # =====================================================================
    # Report view
    # =====================================================================
    with gr.Column(visible=False) as report_view:
        with gr.Row():
            new_scan_btn = gr.Button("← New scan", variant="secondary", size="sm", scale=0)
            download_btn = gr.DownloadButton("↓ Download report (PDF)", size="sm", scale=0)

        # Overall grade
        grade_html = gr.HTML()

        # Modality cards side-by-side, each with its own detail accordion
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                vocal_card_html = gr.HTML()
                with gr.Accordion("Vocal channel details", open=False):
                    vocal_detail_html = gr.HTML()
            with gr.Column(scale=1):
                facial_card_html = gr.HTML()
                with gr.Accordion("Facial channel details", open=False):
                    facial_detail_html = gr.HTML()

        # Cross-modality agreement + narrative + disclaimer
        agreement_html = gr.HTML()
        narrative_md = gr.Markdown()
        gr.HTML(f'<div class="ps-disclaimer">{DISCLAIMER}</div>')

    # Session-scoped state — the last-produced report dict, used by the PDF
    # download so it exports what was actually analyzed (not the mock fixture).
    current_report = gr.State(value=None)

    # ---- Wiring ----------------------------------------------------------
    analyze_btn.click(
        fn=analyze,
        inputs=[vowel_i_input, vowel_o_input, vowel_u_input, pataka_input, smile_input],
        outputs=[
            upload_view,
            processing_view,
            report_view,
            processing_html,
            grade_html,
            vocal_card_html,
            vocal_detail_html,
            facial_card_html,
            facial_detail_html,
            agreement_html,
            narrative_md,
            current_report,
            download_btn,  # PDF path pre-set in final yield → single-click download
        ],
        show_progress="hidden",  # we render our own progress bar in processing_html via staged yields
    )

    new_scan_btn.click(
        fn=new_scan,
        inputs=None,
        outputs=[
            upload_view,
            processing_view,
            report_view,
            vowel_i_input,
            vowel_o_input,
            vowel_u_input,
            pataka_input,
            smile_input,
            current_report,
            download_btn,
        ],
    )

    # NOTE: no separate download_btn.click wiring — the PDF is pre-generated in
    # analyze() and its path is set as download_btn's value in the final yield.
    # Clicking the button then downloads directly (no round-trip).


if __name__ == "__main__":
    # share=True spins up a Gradio tunnel — fine locally, but on Cloud Run the
    # tunnel binary can't reach the outbound relay and startup hangs. Dockerfile
    # sets GRADIO_SHARE=false so the container serves directly on $PORT.
    share = os.environ.get("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
    demo.launch(css=CSS, theme=theme, share=share)
