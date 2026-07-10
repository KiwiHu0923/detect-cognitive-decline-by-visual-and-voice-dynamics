"""ParkScreen — Gradio UI shell.

Pure frontend: no pipeline, no model loading, no Claude call. The Analyze
button drives a mocked 6-stage progress and renders a fixed MOCK_REPORT so
the layout, states, and copy can be iterated on independently of the
Day 3–5 backend work. Replace `analyze()` with the real pipeline once
`src/pipeline.py` lands.

Flow: upload view → click Analyze → progress bar → report view (with a
"New scan" button to return). Report is grouped as a two-tier hierarchy —
overall risk grade (A/B/C), then Vocal and Facial modality cards, each
with a collapsible detail accordion for its per-channel breakdown.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from typing import Any

import gradio as gr

from demo.report_pdf import build_report_pdf


# ---------------------------------------------------------------------------
# Mock data — the only source of truth for what the UI renders. Tweak here
# to preview different report states without touching layout code.
# ---------------------------------------------------------------------------

MOCK_REPORT: dict[str, Any] = {
    "risk_grade": "B",                   # A (low) | B (moderate) | C (elevated)
    "fused_score": 0.68,                 # 0–1
    "modalities": {
        "vocal": {
            "score": 0.70,
            "weight": 0.85,              # weight in overall fused score
            "summary": "Speech channels show mild changes consistent with early phonatory instability.",
            "channels": {
                "phonation": {
                    "status": "ok",       # ok | na_off_task | low_signal
                    "score": 0.72,
                    "confidence": 0.81,
                    "weight": 0.59,       # weight within Vocal
                    "note": "Mildly elevated jitter; reduced HNR",
                },
                "ddk": {
                    "status": "ok",
                    "score": 0.65,
                    "confidence": 0.74,
                    "weight": 0.41,
                    "note": "Slowed rate; mild timing irregularity",
                },
            },
        },
        "facial": {
            "score": 0.54,
            "weight": 0.15,
            "summary": "Reduced smile expressivity; blink rate and head motion within normal range.",
            "channels": {
                "smile": {
                    "status": "ok",
                    "score": 0.54,
                    "confidence": 0.62,
                    "weight": 1.00,
                    "note": "Reduced AU12 amplitude on smile cue",
                },
            },
        },
    },
    "agreement": "partial",             # agree | partial | disagree
    "narrative": """### Key observations

**Phonation.** Sustained-vowel analysis shows mildly elevated jitter (local: 1.4%) and a reduced harmonics-to-noise ratio (HNR: 12.3 dB), consistent with early phonatory instability. F0 range is compressed relative to age-matched controls.

**Articulation (DDK).** /pa-ta-ka/ repetition rate is 5.2 syllables/sec (age-adjusted expected: 6.0–7.0). Inter-syllable interval coefficient of variation is 0.18, indicating mild timing irregularity. Peak amplitude is stable across the utterance.

**Facial dynamics.** Smile-classifier score is moderate (0.54). AU12 (lip-corner puller) amplitude on the smile cue is 0.28, below the healthy-control median. Blink rate and head movement fall within normal ranges.

### Consistency

Speech channels (phonation, DDK) agree. Facial channel is weaker but directionally consistent.

### Recommended next steps

- Consider follow-up assessment with a neurologist or movement-disorder specialist.
- Repeat the same task set in 3–6 months to track any longitudinal change.
""",
}


RECORDING_PROTOCOL = """Record a **single video** containing these tasks, separated by brief pauses:

1. **Sustained /a/** — hold the vowel steady for ~5 seconds.
2. **Rapid /pa-ta-ka/** — repeat as fast and evenly as you can for ~5 seconds.
3. **Face clearly visible** throughout, with a brief neutral → smile → neutral sequence toward the end.

All tasks are language-neutral.
"""


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


PIPELINE_STAGES = [
    ("Extracting audio", 0.5),
    ("Segmenting tasks", 0.5),
    ("Analyzing phonation", 0.7),
    ("Analyzing articulation (DDK)", 0.7),
    ("Analyzing facial dynamics", 0.8),
    ("Generating report", 0.8),
]


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
# Event handlers
# ---------------------------------------------------------------------------

def analyze(video: str | None, progress: gr.Progress = gr.Progress()):
    # Mock mode: video is ignored. Add input validation when wiring the real
    # pipeline in src/pipeline.py.
    total = len(PIPELINE_STAGES)
    for i, (desc, dur) in enumerate(PIPELINE_STAGES):
        progress(i / total, desc=desc)
        time.sleep(dur)
    progress(1.0, desc="Done")

    return (
        gr.update(visible=False),                          # upload_view
        gr.update(visible=True),                           # report_view
        render_grade_badge(MOCK_REPORT),                   # grade_html
        render_modality_card("vocal", MOCK_REPORT["modalities"]["vocal"]),
        render_channels_of(MOCK_REPORT["modalities"]["vocal"]),
        render_modality_card("facial", MOCK_REPORT["modalities"]["facial"]),
        render_channels_of(MOCK_REPORT["modalities"]["facial"]),
        render_agreement(MOCK_REPORT),                     # agreement_html
        MOCK_REPORT["narrative"],                          # narrative_md
    )


def new_scan():
    """Return the UI to the upload view, clearing any prior report state."""
    return (
        gr.update(visible=True),   # upload_view
        gr.update(visible=False),  # report_view
        None,                      # video_input (clear)
    )


def prepare_pdf_download() -> str:
    """Render MOCK_REPORT to a PDF in a temp dir and return the path for
    gr.DownloadButton to serve. Placing the file inside a fresh temp directory
    (rather than an anonymous temp file) preserves the friendly filename in
    the browser download dialog."""
    tmpdir = tempfile.mkdtemp(prefix="parkscreen_")
    path = os.path.join(tmpdir, "parkscreen_report.pdf")
    with open(path, "wb") as f:
        build_report_pdf(MOCK_REPORT, f)
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
    # Upload view
    # =====================================================================
    with gr.Column(visible=True) as upload_view:
        video_input = gr.Video(
            label="Upload video",
            sources=["upload"],
            height=320,
        )
        with gr.Accordion("Recording protocol", open=False):
            gr.Markdown(RECORDING_PROTOCOL)
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

    # ---- Wiring ----------------------------------------------------------
    analyze_btn.click(
        fn=analyze,
        inputs=[video_input],
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
        ],
    )

    new_scan_btn.click(
        fn=new_scan,
        inputs=None,
        outputs=[upload_view, report_view, video_input],
    )

    download_btn.click(
        fn=prepare_pdf_download,
        inputs=None,
        outputs=download_btn,
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=theme)
