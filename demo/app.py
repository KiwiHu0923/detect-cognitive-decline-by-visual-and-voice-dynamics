"""ParkScreen — Gradio UI shell.

Pure frontend: no pipeline, no model loading, no Claude call. The Analyze
button drives a mocked 6-stage progress and renders a fixed MOCK_REPORT so
the layout, states, and copy can be iterated on independently of the
Day 3–5 backend work. Replace `analyze()` with the real pipeline once
`src/pipeline.py` lands.
"""

from __future__ import annotations

import time
from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Mock data — the only source of truth for what the UI renders. Tweak here
# to preview different report states without touching layout code.
# ---------------------------------------------------------------------------

MOCK_REPORT: dict[str, Any] = {
    "risk_level": "Elevated",           # Low | Moderate | Elevated
    "fused_score": 0.68,                # 0–1
    "channels": {
        "phonation": {
            "weight": 0.50,
            "status": "na_off_task",             # ok | na_off_task | low_signal
            "score": 0.72,
            "confidence": 0.81,
            "note": "Mildly elevated jitter; reduced HNR",
        },
        "ddk": {
            "weight": 0.35,
            "status": "ok",
            "score": 0.65,
            "confidence": 0.74,
            "note": "Slowed rate; mild timing irregularity",
        },
        "prosody": {
            "weight": 0.00,
            "status": "na_off_task",
            "score": None,
            "confidence": None,
            "note": "Reading task not detected in upload",
        },
        "facial": {
            "weight": 0.15,
            "status": "ok",
            "score": 0.54,
            "confidence": 0.62,
            "note": "Reduced AU12 amplitude on smile cue",
        },
    },
    "agreement": "partial",             # agree | partial | disagree
    "narrative": """### Key observations

**Phonation.** Sustained-vowel analysis shows mildly elevated jitter (local: 1.4%) and a reduced harmonics-to-noise ratio (HNR: 12.3 dB), consistent with early phonatory instability. F0 range is compressed relative to age-matched controls.

**Articulation (DDK).** /pa-ta-ka/ repetition rate is 5.2 syllables/sec (age-adjusted expected: 6.0–7.0). Inter-syllable interval coefficient of variation is 0.18, indicating mild timing irregularity. Peak amplitude is stable across the utterance.

**Facial dynamics.** Smile-classifier score is moderate (0.54). AU12 (lip-corner puller) amplitude on the smile cue is 0.28, below the healthy-control median. Blink rate and head movement fall within normal ranges.

### Consistency

Speech channels (phonation, DDK) agree. Facial channel is weaker but directionally consistent. **Prosody channel not scored** — reading task not detected in the uploaded video.

### Recommended next steps

- Consider follow-up assessment with a neurologist or movement-disorder specialist.
- Re-record with a short reading passage to enable the prosody channel.
- Repeat the same task set in 3–6 months to track any longitudinal change.
""",
}


RECORDING_PROTOCOL = """Record a **single video** containing these tasks, separated by brief pauses:

1. **Sustained /a/** — hold the vowel steady for ~5 seconds.
2. **Rapid /pa-ta-ka/** — repeat as fast and evenly as you can for ~5 seconds.
3. *(Optional)* **Short reading passage** — a few sentences aloud.
4. **Face clearly visible** throughout, with a brief neutral → smile → neutral sequence toward the end.

The vowel, /pa-ta-ka/, and smile tasks are language-neutral. The reading task is optional and carries a mild language caveat if you are not a Spanish speaker.
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
    "prosody": "Prosody",
    "facial": "Facial dynamics",
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
# Kept as pure functions so the UI is trivially previewable in isolation.
# ---------------------------------------------------------------------------

def render_risk_badge(report: dict[str, Any]) -> str:
    level = report["risk_level"]
    score = report["fused_score"]
    cls = f"ps-risk ps-risk--{level.lower()}"
    return f"""
<div class="{cls}">
  <div class="ps-risk__label">Overall risk level</div>
  <div class="ps-risk__value">{level}</div>
  <div class="ps-risk__score">Fused score {score:.2f}</div>
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
    <span class="ps-card__weight">weight {weight_pct}</span>
  </div>
  <div class="ps-card__body">{body}</div>
</div>
"""


def render_channels_grid(report: dict[str, Any]) -> str:
    order = ["phonation", "ddk", "prosody", "facial"]
    cards = "\n".join(_channel_card(k, report["channels"][k]) for k in order)
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
# Analyze handler — fake 6-stage progress, then return MOCK_REPORT.
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
        gr.update(visible=False),                        # empty_state
        gr.update(visible=True),                         # results_col
        render_risk_badge(MOCK_REPORT),                  # risk_html
        render_channels_grid(MOCK_REPORT),               # channels_html
        render_agreement(MOCK_REPORT),                   # agreement_html
        MOCK_REPORT["narrative"],                        # narrative_md
    )


def reset():
    """Return the UI to its pre-analysis empty state."""
    return (
        gr.update(visible=True),   # empty_state
        gr.update(visible=False),  # results_col
        None,                      # video (clear)
    )


# ---------------------------------------------------------------------------
# Styling — clinical palette (white / slate / blue). Kept inline so the shell
# is a single self-contained file for iteration.
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
  max-width: 1200px !important;
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

/* Empty state */
.ps-empty {
  border: 1px dashed var(--ps-border);
  border-radius: 8px;
  padding: 40px 24px;
  text-align: center;
  color: var(--ps-text-faint);
  background: var(--ps-surface);
  font-size: 14px;
}
.ps-empty strong {
  color: var(--ps-text-muted);
}

/* Risk badge — dark tinted panel with saturated accent, not full-bleed color */
.ps-risk {
  border-radius: 10px;
  padding: 20px 24px;
  margin-bottom: 16px;
  border: 1px solid var(--ps-border);
  border-left-width: 4px;
  background: var(--ps-surface);
}
.ps-risk--low       { border-left-color: var(--ps-risk-low);       background: linear-gradient(90deg, rgba(16,185,129,0.10), var(--ps-surface) 60%); }
.ps-risk--moderate  { border-left-color: var(--ps-risk-moderate);  background: linear-gradient(90deg, rgba(245,158,11,0.10), var(--ps-surface) 60%); }
.ps-risk--elevated  { border-left-color: var(--ps-risk-elevated);  background: linear-gradient(90deg, rgba(239,68,68,0.12),  var(--ps-surface) 60%); }
.ps-risk__label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ps-text-faint);
}
.ps-risk__value {
  font-size: 32px;
  font-weight: 600;
  line-height: 1.2;
  margin-top: 4px;
  color: var(--ps-text);
}
.ps-risk--low       .ps-risk__value { color: var(--ps-risk-low); }
.ps-risk--moderate  .ps-risk__value { color: var(--ps-risk-moderate); }
.ps-risk--elevated  .ps-risk__value { color: var(--ps-risk-elevated); }
.ps-risk__score {
  font-size: 13px;
  color: var(--ps-text-muted);
  margin-top: 6px;
}

/* Channel grid */
.ps-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.ps-card {
  background: var(--ps-surface);
  border: 1px solid var(--ps-border);
  border-radius: 8px;
  padding: 14px 16px;
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
  margin-bottom: 10px;
}
.ps-card__title {
  font-size: 14px;
  font-weight: 600;
  color: var(--ps-text);
}
.ps-card__weight {
  font-size: 11px;
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
  background: var(--ps-surface-2);
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
  margin-bottom: 16px;
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

    with gr.Row(equal_height=False):
        # ---- Left: upload + protocol ---------------------------------------
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Upload video",
                sources=["upload"],
                height=280,
            )
            with gr.Accordion("Recording protocol", open=False):
                gr.Markdown(RECORDING_PROTOCOL)
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary", scale=2)
                reset_btn = gr.Button("Reset", variant="secondary", scale=1)

        # ---- Right: results (empty state + report) ------------------------
        with gr.Column(scale=2):
            with gr.Column(visible=True) as empty_state:
                gr.HTML(
                    '<div class="ps-empty">Upload a video and click <strong>Analyze</strong> '
                    'to see the report.</div>'
                )

            with gr.Column(visible=False) as results_col:
                risk_html = gr.HTML()
                channels_html = gr.HTML()
                agreement_html = gr.HTML()
                narrative_md = gr.Markdown()
                gr.HTML(f'<div class="ps-disclaimer">{DISCLAIMER}</div>')

    # ---- Wiring -----------------------------------------------------------
    analyze_btn.click(
        fn=analyze,
        inputs=[video_input],
        outputs=[
            empty_state,
            results_col,
            risk_html,
            channels_html,
            agreement_html,
            narrative_md,
        ],
    )

    reset_btn.click(
        fn=reset,
        inputs=None,
        outputs=[empty_state, results_col, video_input],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=theme)
