"""One-shot script — render PDF + PNG snapshots of the three demo reports.

Reads existing pipeline artifacts from ``out/{hc,pd,neurovoz_holdout}_demo/``,
merges ``result.json`` + ``report.md`` into the UI-shaped dict via
``demo.app._map_pipeline_to_report``, calls ``build_report_pdf`` to write a
PDF, then shells out to ``pdftoppm`` to rasterize the PDF to PNG. Output
lands in ``demo/screenshots/``.

Run: ``python -m scripts.render_demo_pdfs``
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from demo.app import _map_pipeline_to_report
from demo.report_pdf import build_report_pdf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "out"
SCREEN_DIR = REPO_ROOT / "demo" / "screenshots"

CASES = ["hc_demo", "pd_demo", "neurovoz_holdout_demo"]


def render(case: str) -> None:
    case_dir = OUT_DIR / case
    result = json.loads((case_dir / "result.json").read_text())
    result["report"] = (case_dir / "report.md").read_text()

    report = _map_pipeline_to_report(result)

    pdf_path = SCREEN_DIR / f"{case}_report.pdf"
    with pdf_path.open("wb") as f:
        build_report_pdf(report, f)

    # Rasterize to PNG at 150 DPI (readable for a screenshot deliverable).
    png_prefix = SCREEN_DIR / f"{case}_report"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "150", str(pdf_path), str(png_prefix)],
        check=True,
    )

    pngs = sorted(SCREEN_DIR.glob(f"{case}_report-*.png"))
    print(f"[{case}] pdf={pdf_path.name} pages={len(pngs)}")
    for p in pngs:
        print(f"    {p.name}")


def main() -> None:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        render(case)


if __name__ == "__main__":
    main()
