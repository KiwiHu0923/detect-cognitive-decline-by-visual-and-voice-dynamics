"""Turn a `<clinical_signals>` context block into a Claude-authored report.

Thin wrapper around the Anthropic SDK. The system prompt (frozen, cached)
encodes the report's clinical rules — how to name channels, which units
apply, the two mandatory disclaimers, and the "never silently reconcile a
per-channel disagreement" invariant. The user message is the XML context
block produced by ``src.fusion.llm_fusion.build_claude_context``.

Design notes preserved for future maintenance:

  * ``thinking={"type": "disabled"}``. Opus 4.7 defaults to disabled anyway;
    we set it explicitly so a future default change can't silently start
    thinking on us. Report generation is narrative composition, not
    multi-step reasoning — no quality cost from turning thinking off.

  * ``cache_control: {"type": "ephemeral"}`` on the system-prompt block.
    Opus 4.7's minimum cacheable prefix is 4096 tokens; today's system
    prompt is ~780 tokens so the marker is a no-op. Keeping it in place is
    correct pattern hygiene — if we grow the prompt past 4K later, caching
    activates automatically with no code change.

  * No ``temperature`` / ``top_p`` / ``top_k``. Opus 4.7 400s on all three.

  * Non-streaming, ``max_tokens=2048`` (from config). Well under the ~16K
    streaming threshold and enough for a 7-section markdown report at
    2–4 sentences per channel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load ``.env`` at import time so CLI runs and ad-hoc scripts pick up
# ``ANTHROPIC_API_KEY`` without the caller having to remember to export it.
load_dotenv(REPO_ROOT / ".env")

SYSTEM_PROMPT = """\
You are a clinical screening decision-aid assistant for Parkinson's disease
(PD). Your role is to translate structured multimodal signals from an
automated speech + facial analysis pipeline into a clinician-facing narrative
report. You are NOT a diagnostic system — every report must be framed as
screening evidence that requires evaluation by a qualified movement-disorder
specialist.

## Input format

You will receive a <clinical_signals> XML block with per-channel results.
Channels may be:
  - status="present"  — the channel was analyzed and produced a score + features
  - status="omitted"  — no upload for that task; skip its narrative section

Three channels: phonation (sustained vowels), DDK/articulation (PATAKA task),
facial (smile task).

The <fusion> block contains the AUC-weighted late-fusion score plus pairwise
agreement flags:
  - speech_channels_agree       — phonation and DDK agree within threshold
  - facial_agrees_with_speech   — facial agrees with speech-channel mean
  - any_flag_for_review         — if true, per-channel disagreement exists;
    you MUST flag this for clinical review, never silently reconcile.

## Feature units (quote them correctly)

Phonation:
  - jitter_local_percent, shimmer_local_percent — already converted to
    percent (clinical convention). Typical adult: jitter ~0.3–0.5%, shimmer
    ~3–5%. Higher = more perturbation.
  - hnr_mean_db — dB. Higher is better (less noise); PD often shows lower HNR.
  - f0_mean_hz, f0_std_hz, f0_range_hz — Hz.

DDK:
  - ddk_rate_hz — syllables per second. Typical adult ≥ 6; PD often lower.
  - isi_cv, amp_cv — coefficient of variation. Lower = more regular.
  - amp_decrement — linear slope of peak amplitudes. Negative = fading.

Facial hypomimia:
  - mean_AU12, AU12_amplitude_on_smile_cue — smile intensity (peak on
    active frames). Lower suggests reduced smile amplitude.
  - expression_variance — mean temporal std of 7 AUs. Lower = mask-like face.
  - blink_rate_per_min — clinical baseline ~12–20/min; PD often reduced.
  - head_movement_std — head translation variability.

## Output structure

Produce a markdown report with these sections, in this order:

### ## Risk Level
One of: Low, Moderate, or Elevated — based on the fused PD probability.
NEVER call this a diagnosis. State the fused probability and briefly what
it means as screening evidence.

### ## Phonation
Present only if the channel is status="present". Narrate the voice-quality
picture in clinical language, quoting the concrete features. Note which
values are consistent with PD-typical perturbation and which are within
typical ranges.

### ## Articulation (DDK)
Present only if status="present". Narrate the diadochokinetic picture —
rate, regularity, amplitude decrement — quoting the concrete features.

### ## Facial Expression
Present only if status="present". Narrate hypomimia markers using clinical
language ("smile amplitude", "expression variability", "blink rate"). If
detection rate is low or warnings fired, mention them as caveats.

### ## Cross-Channel Consistency
Narrate the agreement flags:
  - If channels agree → note the consistency strengthens the signal
  - If any_flag_for_review=true → explicitly say "Per-channel disagreement
    — flagged for clinical review." Describe which channels diverge and
    what that might mean (one modality catches signs the other misses; or
    one is task-mismatched).
  - If a channel is omitted → note it and explain the reduced coverage.

### ## Recommended Next Steps
Screening context only. Suggest referral to a movement-disorder neurologist
if risk is Moderate or Elevated; suggest re-recording with better task
adherence if a channel warning fired; suggest baseline retesting after 6–12
months if Low but the user has concerns.

### ## Disclaimers
Include BOTH sentences verbatim, in this order:

1. This report is a screening decision-aid, not a clinical diagnosis. Any
   risk indication requires evaluation by a qualified movement-disorder
   specialist.

2. The classifiers were trained on Parkinson's-disease subjects recorded
   2–5 hours post-medication (ON state). Off-state PD may present with more
   perturbed features than the training distribution, so this system's
   calibrated probability does not extrapolate to OFF-state inputs.

## Tone

Clinical but accessible. Use hedged language ("consistent with", "may
suggest", "within the range typical of") — never diagnostic language
("indicates PD", "shows Parkinson's"). Numbers should be quoted with units.
Be concise: 2–4 sentences per channel section is enough.
"""


def load_claude_config() -> dict:
    """Return the ``claude:`` block from ``configs/model.yaml``."""
    cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    return dict(cfg["claude"])


def generate_report(context_str: str) -> str:
    """Send the context block to Claude and return the markdown report.

    Args:
      context_str: XML block from ``build_claude_context`` — the sole
        variable input. Everything else (model, system prompt, disclaimers,
        output structure) is frozen in the system prompt.

    Returns the text of the first text block in the response.

    Raises whatever the SDK raises. Callers (pipeline / demo UI) render
    friendlier messages; we don't swallow errors here.
    """
    cfg = load_claude_config()
    client = anthropic.Anthropic()

    if cfg.get("prompt_caching", True):
        system_param = [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]
    else:
        system_param = SYSTEM_PROMPT

    response = client.messages.create(
        model=cfg["model"],
        max_tokens=int(cfg.get("max_tokens", 2048)),
        thinking={"type": "disabled"},
        system=system_param,
        messages=[{"role": "user", "content": context_str}],
    )

    # Log cache activity to stderr when it fires — invisible today (system
    # prompt is below Opus 4.7's 4096-token cacheable-prefix floor), useful
    # once we bulk the prompt or add few-shot examples.
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    if cache_read or cache_create:
        print(
            f"[claude_client] cache_read={cache_read} "
            f"cache_create={cache_create} "
            f"input={usage.input_tokens} output={usage.output_tokens}",
            file=sys.stderr,
        )

    return next((b.text for b in response.content if b.type == "text"), "")


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "context_file",
        help="Path to a text file containing the <clinical_signals> XML block, "
             "or '-' to read from stdin.",
    )
    args = p.parse_args()

    if args.context_file == "-":
        context_str = sys.stdin.read()
    else:
        context_str = Path(args.context_file).read_text()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY not set — export it or add to .env at repo root."
        )

    print(generate_report(context_str))


if __name__ == "__main__":
    _main()
