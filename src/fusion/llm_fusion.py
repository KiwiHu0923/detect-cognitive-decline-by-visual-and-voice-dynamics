"""Fuse per-channel PD scores and format the result as a Claude context string.

Two functions, cleanly split:

  * ``fuse_scores`` — the canonical late-fusion mechanic: AUC-excess weight
    renormalization over channels that are actually present, plus pairwise
    agreement flags. Used by ``quick_score.py`` (no LLM) and by
    ``pipeline.py`` (feeds ``build_claude_context``).

  * ``build_claude_context`` — pure presentation: composes an XML-tagged block
    that Claude parses cleanly. Applies the one unit convention we care about
    (jitter/shimmer × 100 → percent, per the NeuroVoz cross-check finding
    2026-07-08) at the presentation boundary so ratio units live inside the
    codebase and clinical units live in the Claude prompt.

Feature curation for narrative: this module only forwards the clinician-facing
subset of each channel's feature dict to Claude. The classifier still sees all
12 phonation + 8 DDK features — those are what determined the score. The
subset shown to Claude covers the vocabulary a clinician uses in a hypomimia /
motor-speech write-up (jitter, shimmer, HNR, F0 stats; DDK rate + regularity +
amplitude decrement) without dumping every collinear jitter variant into the
prompt.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Feature-name curation for Claude narrative. Ordering here defines the order
# Claude sees them in the prompt.
PHONATION_NARRATIVE_FEATURES: tuple[str, ...] = (
    "jitter_local",
    "shimmer_local",
    "hnr_mean_db",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
)
DDK_NARRATIVE_FEATURES: tuple[str, ...] = (
    "ddk_rate_hz",
    "isi_cv",
    "amp_cv",
    "amp_decrement",
)

# Ratio-unit features that must be × 100 before Claude sees them (clinical
# convention is percent; we store ratios internally).
_RATIO_TO_PERCENT: dict[str, str] = {
    "jitter_local": "jitter_local_percent",
    "jitter_rap": "jitter_rap_percent",
    "jitter_ppq5": "jitter_ppq5_percent",
    "shimmer_local": "shimmer_local_percent",
    "shimmer_apq5": "shimmer_apq5_percent",
    "shimmer_apq11": "shimmer_apq11_percent",
    "shimmer_dda": "shimmer_dda_percent",
}


def load_fusion_config() -> tuple[dict[str, float], float]:
    """Return ``(weights, agreement_threshold)`` from ``configs/model.yaml``."""
    cfg = yaml.safe_load((REPO_ROOT / "configs/model.yaml").read_text())
    fusion = cfg["fusion"]
    return dict(fusion["weights"]), float(fusion.get("agreement_threshold", 0.30))


def fuse_scores(
    scores: dict[str, float | None],
    weights: dict[str, float],
    agreement_threshold: float = 0.30,
) -> dict:
    """Weighted late fusion over channels present, plus agreement flags.

    Args:
      scores: ``{"phonation": prob | None, "ddk": prob | None, "facial": prob | None}``
        — None means the channel wasn't uploaded / was withheld by a quality gate.
      weights: fusion weights (from ``configs/model.yaml``); renormalized over
        the channels that are actually present.
      agreement_threshold: pairwise probability gap under which two channels
        are considered to agree. Loaded from config; default 0.30 matches the
        model.yaml value.

    Returns a dict:
      ``fused_score``           float in [0,1] or None if no channel present
      ``weights_normalized``    ``{channel: renormalized_weight}`` (only present channels)
      ``channels_used``         list[str] of channels that contributed
      ``agreement``             ``{
                                    "speech_channels_agree": bool | None,
                                    "facial_agrees_with_speech": bool | None,
                                    "any_flag_for_review": bool,
                                }``
        Agreement fields are None when the comparison is undefined (e.g. only
        one speech channel present).
    """
    present = {c: s for c, s in scores.items() if s is not None}
    if not present:
        return {
            "fused_score": None,
            "weights_normalized": {},
            "channels_used": [],
            "agreement": {
                "speech_channels_agree": None,
                "facial_agrees_with_speech": None,
                "any_flag_for_review": False,
            },
        }

    w = {c: float(weights.get(c, 0.0)) for c in present}
    total = sum(w.values())
    if total == 0:
        # Config has zero weight for every present channel — fall back to
        # equal weighting so we don't crash. Rare edge case.
        w = {c: 1.0 for c in present}
        total = float(len(present))
    w_norm = {c: v / total for c, v in w.items()}
    fused = sum(w_norm[c] * present[c] for c in present)

    phon = present.get("phonation")
    ddk = present.get("ddk")
    facial = present.get("facial")

    if phon is not None and ddk is not None:
        speech_agree: bool | None = abs(phon - ddk) < agreement_threshold
    else:
        speech_agree = None

    # Facial vs speech: compare against the mean of whatever speech channels
    # are present. If no speech channels, comparison is undefined.
    speech_probs = [p for p in (phon, ddk) if p is not None]
    if facial is not None and speech_probs:
        speech_mean = sum(speech_probs) / len(speech_probs)
        facial_agree: bool | None = abs(facial - speech_mean) < agreement_threshold
    else:
        facial_agree = None

    any_flag = (speech_agree is False) or (facial_agree is False)

    return {
        "fused_score": float(fused),
        "weights_normalized": w_norm,
        "channels_used": list(present.keys()),
        "agreement": {
            "speech_channels_agree": speech_agree,
            "facial_agrees_with_speech": facial_agree,
            "any_flag_for_review": bool(any_flag),
        },
    }


def _fmt_num(x: float | int | None, digits: int = 3) -> str:
    """Format a float for the Claude prompt; NaN/None → 'null'."""
    if x is None:
        return "null"
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return "null"
    if fx != fx:  # NaN
        return "null"
    return f"{fx:.{digits}f}"


def _to_percent(features: dict[str, float]) -> dict[str, float]:
    """Apply the ratio → percent conversion for jitter/shimmer keys.

    Non-ratio features pass through unchanged. Missing keys are silently
    skipped — the narrative-feature filter downstream handles absence.
    """
    out: dict[str, float] = {}
    for k, v in features.items():
        if k in _RATIO_TO_PERCENT:
            new_k = _RATIO_TO_PERCENT[k]
            out[new_k] = None if v is None else float(v) * 100.0
        else:
            out[k] = v
    return out


def _render_features_block(
    features: dict[str, float | None],
    ordered_keys: tuple[str, ...],
    indent: str,
) -> str:
    """Emit ``key: value`` lines for the ordered keys that exist in features."""
    lines: list[str] = []
    for k in ordered_keys:
        if k not in features:
            continue
        # HNR gets one decimal (dB scale); rate/CV/decrement three; F0 rounds to Hz.
        if k.endswith("_hz") and k.startswith("f0"):
            digits = 1
        elif k.endswith("_db"):
            digits = 2
        elif k == "amp_decrement":
            digits = 4
        else:
            digits = 3
        lines.append(f"{indent}{k}: {_fmt_num(features[k], digits)}")
    return "\n".join(lines)


def _render_phonation(channel: dict | None) -> str:
    if channel is None:
        return '  <phonation status="omitted" reason="no vowel upload"/>'
    score = channel.get("score")
    raw_features: dict = channel.get("features", {}) or {}
    features = _to_percent(raw_features)
    # Feature keys to show: rename ratio keys to their _percent versions.
    display_keys = tuple(
        _RATIO_TO_PERCENT.get(k, k) for k in PHONATION_NARRATIVE_FEATURES
    )
    feat_block = _render_features_block(features, display_keys, indent="      ")
    return (
        '  <phonation status="present">\n'
        f"    pd_probability: {_fmt_num(score)}\n"
        "    features:\n"
        f"{feat_block}\n"
        "  </phonation>"
    )


def _render_ddk(channel: dict | None) -> str:
    if channel is None:
        return '  <ddk status="omitted" reason="no pataka upload"/>'
    score = channel.get("score")
    features: dict = channel.get("features", {}) or {}
    feat_block = _render_features_block(features, DDK_NARRATIVE_FEATURES, indent="      ")
    return (
        '  <ddk status="present">\n'
        f"    pd_probability: {_fmt_num(score)}\n"
        "    features:\n"
        f"{feat_block}\n"
        "  </ddk>"
    )


def _render_facial(channel: dict | None, facial_summary: dict | None) -> str:
    if channel is None:
        return '  <facial status="omitted" reason="no smile upload"/>'
    score = channel.get("score")
    lines = [
        '  <facial status="present">',
        f"    pd_probability: {_fmt_num(score)}",
    ]
    if facial_summary is not None:
        # Hypomimia markers: emit whichever of the five are non-None. The
        # summarize function returns None for fields it couldn't compute.
        marker_keys = (
            "mean_AU12",
            "AU12_amplitude_on_smile_cue",
            "expression_variance",
            "blink_rate_per_min",
            "head_movement_std",
        )
        lines.append("    hypomimia_markers:")
        for k in marker_keys:
            v = facial_summary.get(k)
            lines.append(f"      {k}: {_fmt_num(v)}")
        det = facial_summary.get("detection_rate")
        if det is not None:
            lines.append(f"    detection_rate: {_fmt_num(det, 2)}")
        warnings = facial_summary.get("warnings") or []
        if warnings:
            joined = "; ".join(str(w) for w in warnings)
            lines.append(f'    warnings: "{joined}"')
        else:
            lines.append("    warnings: []")
    lines.append("  </facial>")
    return "\n".join(lines)


def _render_fusion(fusion_result: dict) -> str:
    if fusion_result["fused_score"] is None:
        return '  <fusion status="omitted" reason="no channels available"/>'
    ag = fusion_result["agreement"]
    w = fusion_result["weights_normalized"]
    weights_str = ", ".join(f"{c}: {v:.2f}" for c, v in w.items())

    def bool_or_null(x: bool | None) -> str:
        return "null" if x is None else ("true" if x else "false")

    return (
        "  <fusion>\n"
        f"    fused_pd_probability: {_fmt_num(fusion_result['fused_score'])}\n"
        f"    weights_normalized: {{{weights_str}}}\n"
        f"    speech_channels_agree: {bool_or_null(ag['speech_channels_agree'])}\n"
        f"    facial_agrees_with_speech: {bool_or_null(ag['facial_agrees_with_speech'])}\n"
        f"    any_flag_for_review: {bool_or_null(ag['any_flag_for_review'])}\n"
        "  </fusion>"
    )


def build_claude_context(
    channels: dict[str, dict | None],
    facial_summary: dict | None,
    fusion_result: dict,
) -> str:
    """Compose the ``<clinical_signals>`` XML block for the Claude prompt.

    Args:
      channels: per-channel dicts of the form
        ``{"phonation": {"score": prob, "features": {ratio_units}}, ...}``.
        A channel value of ``None`` renders as ``status="omitted"``.
        Phonation feature values are in *ratio* units — the ratio→percent
        conversion happens inside this function.
      facial_summary: output of ``src.vision.summarize.summarize_facial_features``
        (or None). Only used when ``channels["facial"]`` is present.
      fusion_result: output of ``fuse_scores``.

    Returns the tagged block as a single string, ready to drop into the
    Claude user message.
    """
    parts = [
        "<clinical_signals>",
        _render_phonation(channels.get("phonation")),
        _render_ddk(channels.get("ddk")),
        _render_facial(channels.get("facial"), facial_summary),
        _render_fusion(fusion_result),
        "</clinical_signals>",
    ]
    return "\n".join(parts)
