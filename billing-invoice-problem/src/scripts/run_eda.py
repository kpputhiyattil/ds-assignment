"""
run_eda.py — invoice-focused exploratory data analysis on Invoices_users.parquet.

Run this **before** training so data design choices are documented first.
Only needs ``data/raw/Invoices_users.parquet`` (optionally uses processed events
if already built).

Usage (from project root, venv active):
    python -m src.scripts.run_eda
    # or:
    python -m src.data.eda
    make eda
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow `python src/scripts/run_eda.py` from project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.eda import write_eda_report  # noqa: E402


def main(*, save_plots: bool = True) -> tuple[Path, Path]:
    json_path, md_path = write_eda_report(save_plots=save_plots)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 60)
    print("INVOICE EDA COMPLETE")
    print("=" * 60)
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Figures:  {len(report.get('figures', []))}")
    print(f"Findings: {len(report.get('findings', []))}")
    print("-" * 60)
    for i, finding in enumerate(report.get("findings", []), 1):
        msg = (
            finding["message"][:120]
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
