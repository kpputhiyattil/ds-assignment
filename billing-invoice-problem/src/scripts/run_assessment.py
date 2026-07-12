"""
run_assessment.py — build the single mini-report from pipeline artifacts.

Writes ``reports/assessment/mini_report.md`` (approach, with-vs-without summary,
ablation tables, SHAP, improvements). Run **after** train / ablation / explain.

Usage:
    python -m src.scripts.run_assessment
    make assess
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.reporting.assessment import collect_artifacts, write_assessment_report  # noqa: E402


def main() -> Path:
    arts = collect_artifacts()
    missing = [
        k
        for k, v in arts.items()
        if v is None
        and k not in ("eda", "package_manifest", "batch_summary", "train_metadata")
    ]
    if arts.get("eda") is None:
        print("Note: EDA report missing — run `make eda` first for richer narrative.")
    if missing:
        print(
            "Warning: missing artifacts (report will note gaps): "
            + ", ".join(missing)
        )
    path = write_assessment_report()
    print("\n" + "=" * 60)
    print("MINI-REPORT (single file to send)")
    print("=" * 60)
    print(f"Markdown: {path}")
    print("Copy:     data/artifacts/mini_report.md")
    ab = arts.get("ablation") or {}
    if ab:
        print(f"Verdict:  interval_good_enough={ab.get('interval_good_enough')}")
        print(f"Decision: {ab.get('decision')}")
    print("=" * 60)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
