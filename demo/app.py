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

from demo.report_pdf import build_report_pdf
from src.pipeline import run_pipeline


RECORDING_PROTOCOL = """ParkScreen analyzes **three task-matched recordings** — one bucket per task, all language-neutral. Any bucket may be left empty; that channel is reported as **N/A** and the fusion renormalizes over the channels you did provide.

1. **Sustained vowels /i/, /o/, /u/** — up to 3 reps per vowel, ~3–5 s each, steady pitch, comfortable loudness. Audio only. Name files as `<label>_<VOWEL><REP>.<ext>`, e.g. `HC_I1.m4a`, `PD_O2.wav`, `me_U3.mp3` — the pipeline groups reps by vowel and averages within each vowel before combining. At least one file from {I, O, U} is required to score phonation.
2. **Rapid /pa-ta-ka/ repetition** — 1+ reps of ~5 s each, as fast and evenly as you can. Audio only. Features are averaged across reps.
3. **Smile ×3 alternating with neutral face** — 8–12 s total per clip, each smile phase ~2–3 s + neutral ~1–2 s between. Face clearly visible, per the Islam 2023 protocol. Video with audio ok (audio is ignored). Multiple clips accepted; the pipeline max-pools the classifier score and takes the hypomimia summary from the selected clip.
"""


VOWEL_INSTRUCTIONS = (
    "**Sustained vowels /i/, /o/, /u/** — up to 3 reps per vowel, ~3–5 s each.  "
    "Filenames must end with the vowel letter + rep number (e.g. `HC_I1.m4a`, `me_O2.wav`, `PD_U3.mp3`) — "
    "the pipeline uses the letter to group reps. At least one of {I, O, U} required."
)

PATAKA_INSTRUCTIONS = (
    "**Rapid /pa-ta-ka/ repetition** — 1+ reps of ~5 s each, as fast and evenly as you can. "
    "Multiple files are averaged. Any audio format."
)

SMILE_INSTRUCTIONS = (
    "**Smile ×3 alternating with neutral face** — 8–12 s per clip. Face clearly visible. "
    "Video only. Multiple clips → max-pool across clips."
)


DISCLAIMER = (
    "<strong>Screening decision-aid, not a diagnosis.</strong> "
    "ParkScreen provides supporting evidence for clinical review only. "
    "It is not a substitute for evaluation by a qualified healthcare "
    "professional and must not be used for self-diagnosis."
)


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

def _stage_uploads(files: list | None, dest: Path) -> int:
    """Copy uploaded files into ``dest`` preserving original filenames.

    Gradio hands us paths in its cache dir; the vowel-filter parses the vowel
    letter from filename tokens like ``PD_I1.m4a``, so we preserve that here
    rather than pointing the pipeline at the cache dir directly (safer against
    any future Gradio filename mangling). Returns the number of files copied.
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
        shutil.copy2(src_path, dest / src_path.name)
        n += 1
    return n


def analyze(
    vowel_files: list | None,
    pataka_files: list | None,
    smile_files: list | None,
    progress: gr.Progress = gr.Progress(),
):
    """Stage uploads → ``run_pipeline`` → UI report dict → seven UI updates.

    Returns eight values (matches the ``outputs=[...]`` list on the click
    handler; the last one is the ``current_report`` state used by the PDF
    button).
    """
    if not any([vowel_files, pataka_files, smile_files]):
        raise gr.Error("Upload at least one bucket (vowel, PATAKA, or smile) before analyzing.")

    progress(0.05, desc="Staging uploads")
    workdir = Path(tempfile.mkdtemp(prefix="parkscreen_ui_"))
    vowel_dir = workdir / "vowel"
    pataka_dir = workdir / "pataka"
    smile_dir = workdir / "smile"

    n_vowel = _stage_uploads(vowel_files, vowel_dir)
    n_pataka = _stage_uploads(pataka_files, pataka_dir)
    n_smile = _stage_uploads(smile_files, smile_dir)

    progress(0.15, desc=f"Running phonation ({n_vowel} files)")
    # run_pipeline is blocking; there's no per-channel progress hook to
    # interpolate between. Nudge the bar between the big stages instead.
    # Any channel with 0 files is passed as None so run_pipeline reports it
    # as N/A rather than crashing on an empty directory.
    result = run_pipeline(
        vowel_dir=vowel_dir if n_vowel else None,
        pataka_dir=pataka_dir if n_pataka else None,
        smile_dir=smile_dir if n_smile else None,
        call_claude=True,
    )

    progress(0.9, desc="Rendering report")
    report = _map_pipeline_to_report(result)
    progress(1.0, desc="Done")

    return (
        gr.update(visible=False),                                    # upload_view
        gr.update(visible=True),                                     # report_view
        render_grade_badge(report),                                  # grade_html
        render_modality_card("vocal", report["modalities"]["vocal"]),
        render_channels_of(report["modalities"]["vocal"]),
        render_modality_card("facial", report["modalities"]["facial"]),
        render_channels_of(report["modalities"]["facial"]),
        render_agreement(report),                                    # agreement_html
        report["narrative"],                                         # narrative_md
        report,                                                      # current_report state
    )


def new_scan():
    """Return the UI to the upload view, clearing any prior report state."""
    return (
        gr.update(visible=True),   # upload_view
        gr.update(visible=False),  # report_view
        None,                      # vowel_input (clear)
        None,                      # pataka_input (clear)
        None,                      # smile_input (clear)
        None,                      # current_report state (clear)
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
    with gr.Column(visible=True) as upload_view:
        with gr.Accordion("Recording protocol", open=True):
            gr.Markdown(RECORDING_PROTOCOL)

        gr.Markdown("### 1 · Sustained vowels")
        gr.Markdown(VOWEL_INSTRUCTIONS)
        vowel_input = gr.File(
            label="Vowel recordings (audio)",
            file_count="multiple",
            file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
            height=140,
        )

        gr.Markdown("### 2 · Rapid /pa-ta-ka/ (PATAKA)")
        gr.Markdown(PATAKA_INSTRUCTIONS)
        pataka_input = gr.File(
            label="PATAKA recordings (audio)",
            file_count="multiple",
            file_types=["audio", ".m4a", ".mp3", ".wav", ".flac", ".aac"],
            height=140,
        )

        gr.Markdown("### 3 · Smile task")
        gr.Markdown(SMILE_INSTRUCTIONS)
        smile_input = gr.File(
            label="Smile videos",
            file_count="multiple",
            file_types=["video", ".mp4", ".mov", ".avi", ".mkv"],
            height=140,
        )

        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

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
        inputs=[vowel_input, pataka_input, smile_input],
        outputs=[
            upload_view,
            report_view,
            grade_html,
            vocal_card_html,
            vocal_detail_html,
            facial_card_html,
            facial_detail_html,
            agreement_html,
            narrative_md,
            current_report,
        ],
    )

    new_scan_btn.click(
        fn=new_scan,
        inputs=None,
        outputs=[upload_view, report_view, vowel_input, pataka_input, smile_input, current_report],
    )

    download_btn.click(
        fn=prepare_pdf_download,
        inputs=[current_report],
        outputs=download_btn,
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=theme)
