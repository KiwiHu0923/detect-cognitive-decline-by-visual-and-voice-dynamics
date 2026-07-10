"""ParkScreen PDF report builder — clinical light theme.

Called by demo/app.py's download button. Uses reportlab for layout and
matplotlib (headless Agg backend) for the modality donut charts. Palette
is deliberately light/white — appropriate for a printed clinical report,
even though the web app itself is dark.
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Palette (light clinical)
# ---------------------------------------------------------------------------

TEXT = colors.HexColor("#0f172a")
TEXT_MUTED = colors.HexColor("#475569")
TEXT_FAINT = colors.HexColor("#94a3b8")
SURFACE = colors.HexColor("#f8fafc")
BORDER = colors.HexColor("#e5e7eb")
RISK_LOW = colors.HexColor("#059669")
RISK_MODERATE = colors.HexColor("#d97706")
RISK_ELEVATED = colors.HexColor("#dc2626")

GRADE_COLORS = {"A": RISK_LOW, "B": RISK_MODERATE, "C": RISK_ELEVATED}

# hex strings for matplotlib (no HexColor wrappers)
_C_TRACK = "#e5e7eb"
_C_LOW = "#059669"
_C_MOD = "#d97706"
_C_ELE = "#dc2626"


# ---------------------------------------------------------------------------
# Labels + copy (duplicated from app.py to keep this module self-contained)
# ---------------------------------------------------------------------------

GRADE_LABELS = {
    "A": "Low likelihood of features consistent with PD",
    "B": "Moderate likelihood of features consistent with PD",
    "C": "Elevated likelihood of features consistent with PD",
}

CHANNEL_LABELS = {
    "phonation": "Phonation",
    "ddk": "Articulation (DDK)",
    "smile": "Smile task",
}

MODALITY_LABELS = {"vocal": "Vocal", "facial": "Facial"}

AGREEMENT_COPY = {
    "agree": (
        "Channels agree",
        "All active channels point in the same direction. Higher confidence in the fused score.",
    ),
    "partial": (
        "Channels partially agree",
        "Speech channels align; facial signal is weaker but directionally consistent.",
    ),
    "disagree": (
        "Channels disagree",
        "Active channels point in different directions. Flag for clinical review.",
    ),
}

AGREEMENT_COLORS = {
    "agree": RISK_LOW,
    "partial": RISK_MODERATE,
    "disagree": RISK_ELEVATED,
}

DISCLAIMER_TEXT = (
    "<b>Screening decision-aid, not a diagnosis.</b> "
    "ParkScreen provides supporting evidence for clinical review only. "
    "It is not a substitute for evaluation by a qualified healthcare "
    "professional and must not be used for self-diagnosis."
)


# ---------------------------------------------------------------------------
# Donut chart via matplotlib
# ---------------------------------------------------------------------------

def _severity_hex(score: float) -> str:
    if score < 0.4:
        return _C_LOW
    if score < 0.7:
        return _C_MOD
    return _C_ELE


def _make_donut_png(score: float, size_in: float = 1.5) -> BytesIO:
    """Render a matplotlib donut for `score` in [0,1]. Returns BytesIO PNG."""
    color = _severity_hex(score)
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=200)
    ax.pie(
        [max(score, 1e-4), max(1 - score, 1e-4)],
        colors=[color, _C_TRACK],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.5),
    )
    ax.text(
        0, 0.08, f"{int(round(score * 100))}",
        ha="center", va="center",
        fontsize=32, fontweight="bold", color="#0f172a",
    )
    ax.text(
        0, -0.32, "/ 100",
        ha="center", va="center",
        fontsize=8, color="#94a3b8",
    )
    ax.set(aspect="equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    buf = BytesIO()
    plt.savefig(
        buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0.05
    )
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

STYLES = {
    "title": ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=20,
        textColor=TEXT, leading=24, spaceAfter=4,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=10,
        textColor=TEXT_MUTED, leading=14, spaceAfter=18,
    ),
    "section_h": ParagraphStyle(
        "sec", fontName="Helvetica-Bold", fontSize=12,
        textColor=TEXT, leading=16, spaceBefore=14, spaceAfter=6,
    ),
    "grade_letter": ParagraphStyle(
        "gl", fontName="Helvetica-Bold", fontSize=54,
        alignment=1, leading=58,
    ),
    "grade_label": ParagraphStyle(
        "glab", fontName="Helvetica", fontSize=8,
        textColor=TEXT_FAINT, leading=10,
    ),
    "grade_desc": ParagraphStyle(
        "gdesc", fontName="Helvetica", fontSize=13,
        textColor=TEXT, leading=17, spaceAfter=3,
    ),
    "grade_score": ParagraphStyle(
        "gscore", fontName="Helvetica", fontSize=10,
        textColor=TEXT_MUTED, leading=13,
    ),
    "mod_title": ParagraphStyle(
        "mtitle", fontName="Helvetica-Bold", fontSize=12,
        textColor=TEXT, leading=15, alignment=1,
    ),
    "mod_weight": ParagraphStyle(
        "mwt", fontName="Helvetica", fontSize=8,
        textColor=TEXT_FAINT, alignment=1, spaceAfter=6,
    ),
    "mod_summary": ParagraphStyle(
        "msum", fontName="Helvetica-Oblique", fontSize=9,
        textColor=TEXT_MUTED, leading=12, alignment=1,
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10,
        textColor=TEXT, leading=14,
    ),
    "body_muted": ParagraphStyle(
        "bm", fontName="Helvetica", fontSize=9,
        textColor=TEXT_MUTED, leading=12,
    ),
    "table_h": ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=8,
        textColor=TEXT_FAINT, leading=10,
    ),
    "disclaimer": ParagraphStyle(
        "dis", fontName="Helvetica", fontSize=8,
        textColor=TEXT_MUTED, leading=12,
    ),
    "narrative_h": ParagraphStyle(
        "nh", fontName="Helvetica-Bold", fontSize=11,
        textColor=TEXT, leading=14, spaceBefore=8, spaceAfter=4,
    ),
    "narrative_p": ParagraphStyle(
        "np", fontName="Helvetica", fontSize=10,
        textColor=TEXT, leading=14, spaceAfter=4,
    ),
    "narrative_li": ParagraphStyle(
        "nli", fontName="Helvetica", fontSize=10,
        textColor=TEXT, leading=14,
        leftIndent=14, bulletIndent=4,
    ),
}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _grade_section(report: dict[str, Any]):
    grade = report["risk_grade"]
    color = GRADE_COLORS[grade]
    desc = GRADE_LABELS[grade]
    letter_style = ParagraphStyle(
        "gl_col", parent=STYLES["grade_letter"], textColor=color
    )

    left = Paragraph(grade, letter_style)
    right = [
        Paragraph("OVERALL RISK GRADE", STYLES["grade_label"]),
        Paragraph(desc, STYLES["grade_desc"]),
        Paragraph(f"Fused score {report['fused_score']:.2f}", STYLES["grade_score"]),
    ]

    tbl = Table([[left, right]], colWidths=[1.2 * inch, 5.2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("LINEBEFORE", (0, 0), (0, 0), 3, color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    return tbl


def _modality_cell(key: str, mod: dict[str, Any]):
    name = MODALITY_LABELS[key]
    weight_pct = int(round(mod["weight"] * 100))
    donut_buf = _make_donut_png(mod["score"], size_in=1.4)
    img = Image(donut_buf, width=1.4 * inch, height=1.4 * inch)
    return [
        Paragraph(name, STYLES["mod_title"]),
        Paragraph(f"weight {weight_pct}%", STYLES["mod_weight"]),
        img,
        Spacer(1, 4),
        Paragraph(mod["summary"], STYLES["mod_summary"]),
    ]


def _modalities_section(report: dict[str, Any]):
    left = _modality_cell("vocal", report["modalities"]["vocal"])
    right = _modality_cell("facial", report["modalities"]["facial"])

    tbl = Table([[left, right]], colWidths=[3.2 * inch, 3.2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (0, 0), 0.5, BORDER),
        ("BOX", (1, 0), (1, 0), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return tbl


def _channel_row(key: str, ch: dict[str, Any]):
    name = CHANNEL_LABELS[key]
    weight = int(round(ch["weight"] * 100))
    if ch["status"] == "ok":
        score_s = f"{ch['score']:.2f}"
        conf_s = f"{ch['confidence']:.2f}"
    else:
        score_s = "N/A"
        conf_s = "—"

    name_html = (
        f"<b>{name}</b><br/>"
        f"<font color='#94a3b8' size='7'>{weight}% within modality</font>"
    )
    return [
        Paragraph(name_html, STYLES["body"]),
        Paragraph(score_s, STYLES["body"]),
        Paragraph(conf_s, STYLES["body_muted"]),
        Paragraph(ch["note"], STYLES["body_muted"]),
    ]


def _channel_table(mod: dict[str, Any]):
    header = [
        Paragraph("CHANNEL", STYLES["table_h"]),
        Paragraph("SCORE", STYLES["table_h"]),
        Paragraph("CONFIDENCE", STYLES["table_h"]),
        Paragraph("NOTE", STYLES["table_h"]),
    ]
    rows = [header] + [_channel_row(k, v) for k, v in mod["channels"].items()]
    tbl = Table(rows, colWidths=[1.6 * inch, 0.7 * inch, 0.9 * inch, 3.2 * inch])
    tbl.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return tbl


def _agreement_section(report: dict[str, Any]):
    label, body = AGREEMENT_COPY[report["agreement"]]
    color = AGREEMENT_COLORS[report["agreement"]]
    cell = [
        Paragraph(f"<b>{label}</b>", STYLES["body"]),
        Spacer(1, 2),
        Paragraph(body, STYLES["body_muted"]),
    ]
    tbl = Table([[cell]], colWidths=[6.4 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("LINEBEFORE", (0, 0), (0, 0), 3, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def _narrative_flowables(markdown_text: str):
    """Tiny MD → reportlab converter: ### headers, **bold**, - bullets, paragraphs."""
    flow = []
    lines = markdown_text.strip().split("\n")
    bullets: list[str] = []

    def flush_bullets():
        for b in bullets:
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", b)
            flow.append(Paragraph(f"• {content}", STYLES["narrative_li"]))
        bullets.clear()

    for line in lines:
        if line.startswith("### "):
            flush_bullets()
            flow.append(Paragraph(line[4:], STYLES["narrative_h"]))
        elif line.startswith("- "):
            bullets.append(line[2:])
        elif line.strip() == "":
            flush_bullets()
        else:
            flush_bullets()
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            flow.append(Paragraph(content, STYLES["narrative_p"]))
    flush_bullets()
    return flow


def _disclaimer_section():
    tbl = Table(
        [[Paragraph(DISCLAIMER_TEXT, STYLES["disclaimer"])]],
        colWidths=[6.4 * inch],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report_pdf(report: dict[str, Any], out_stream) -> None:
    """Write a PDF representation of `report` to `out_stream` (file-like)."""
    doc = SimpleDocTemplate(
        out_stream,
        pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="ParkScreen Report",
        author="ParkScreen",
    )
    story = [
        Paragraph("ParkScreen — Screening Report", STYLES["title"]),
        Paragraph(
            "Multimodal decision-aid for Parkinson's disease signs — <i>not a diagnosis</i>.",
            STYLES["subtitle"],
        ),
        _grade_section(report),
        Spacer(1, 16),
        Paragraph("Modality scores", STYLES["section_h"]),
        _modalities_section(report),
        Paragraph("Vocal channel details", STYLES["section_h"]),
        _channel_table(report["modalities"]["vocal"]),
        Paragraph("Facial channel details", STYLES["section_h"]),
        _channel_table(report["modalities"]["facial"]),
        Paragraph("Consistency", STYLES["section_h"]),
        _agreement_section(report),
        Paragraph("Clinical narrative", STYLES["section_h"]),
        *_narrative_flowables(report["narrative"]),
        Spacer(1, 18),
        _disclaimer_section(),
    ]
    doc.build(story)
