"""
run_eda.py — run exploratory data analysis on actual Landlords / Companies data.

Usage (from project root, venv active):
    python scripts/run_eda.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.eda import write_eda_report


def main(*, save_plots: bool = True) -> tuple[Path, Path]:
    """Load raw data, run EDA, write versioned report under reports/eda_runs/."""
    json_path, md_path = write_eda_report(save_plots=save_plots)

    with open(json_path, encoding="utf-8") as fp:
        report = json.load(fp)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"Run dir:  {json_path.parent}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Latest:   reports/eda_report.json | reports/eda_report.md")
    print(f"Figures:  {len(report.get('figures', []))} files")
    print(f"Findings: {len(report.get('findings', []))}")
    print("-" * 60)
    for i, finding in enumerate(report.get("findings", []), 1):
        msg = (
            finding["message"][:100]
            .replace("≥", ">=")
            .replace("–", "-")
            .replace("—", "-")
            .replace("≈", "~=")
            .replace("…", "...")
        )
        print(f"  {i}. [{finding['severity']}] {finding['area']}: {msg}")
    print("=" * 60)

    return json_path, md_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
