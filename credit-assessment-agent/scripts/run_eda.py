"""
run_eda.py — company-profile EDA on Companies.parquet.

Run this **before** Streamlit / batch assessment so missingness, status mixes,
and budget-utilisation cutoffs that justify guardrails are documented first.

Usage (from project root, venv active):
    python scripts/run_eda.py
    make eda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import get_settings
from agent.eda import write_eda_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Companies.parquet EDA")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override DATA_PATH from .env",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    data_path = args.data_path or get_settings().data_path

    json_path, md_path = write_eda_report(data_path=data_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 60)
    print("COMPANY EDA COMPLETE")
    print("=" * 60)
    print(f"Data:     {report.get('data_path')}")
    print(f"Rows:     {report.get('n_rows')}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Findings: {len(report.get('findings', []))}")
    print("-" * 60)
    for i, finding in enumerate(report.get("findings", []), 1):
        msg = (
            finding["message"][:140]
            .replace("≥", ">=")
            .replace("–", "-")
            .replace("—", "-")
            .replace("≈", "~=")
            .replace("…", "...")
        )
        print(f"  {i}. [{finding['severity']}] {finding['area']}: {msg}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    raise SystemExit(main())
