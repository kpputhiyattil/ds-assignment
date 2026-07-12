"""
Batch-assess companies and group by recommendation (Continue / Review / No-Go).

Usage
-----
  # Size from .env BATCH_SIZE (full | 50 | 150 | …)
  python scripts/batch_assess.py

  # Override .env for this run
  python scripts/batch_assess.py --limit 50 --offset 0

  # Write detailed JSON results
  python scripts/batch_assess.py --out outputs/batch_report.json

Notes
-----
Each assessment calls the configured LLM (see .env). Set BATCH_SIZE in .env
to control how many companies are assessed when --limit is omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Project root on sys.path when run as `python scripts/batch_assess.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.compat  # noqa: F401 — langchain 0.2+ shim
from agent.config import get_settings
from agent.credit_agent import CreditAgent
from agent.data_loader import DataLoader
from agent.models import Recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy tool warnings during batch runs
logging.getLogger("agent.tools.business_profile").setLevel(logging.ERROR)
logging.getLogger("agent.tools.financial_health").setLevel(logging.ERROR)
logging.getLogger("agent.tools.client_engagement").setLevel(logging.ERROR)
logger = logging.getLogger("batch_assess")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch credit assessment grouped by verdict")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of companies to assess (overrides BATCH_SIZE from .env)",
    )
    p.add_argument("--offset", type=int, default=0, help="Skip this many IDs from the sorted list")
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated CompanyIDs (overrides --limit/--offset)",
    )
    p.add_argument(
        "--demo-three",
        action="store_true",
        help="Assess the three assignment demos: Continue / Review / Guardrail No-Go",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write full JSON results (e.g. outputs/batch_150.json)",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override DATA_PATH (default: from Settings / .env)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    get_settings.cache_clear()
    settings = get_settings()

    data_path = args.data_path or settings.data_path
    loader = DataLoader(data_path)
    all_ids = loader.list_company_ids()

    batch_label = settings.batch_size
    limit = args.limit

    if args.demo_three:
        selected = ["COMPANY_0088", "COMPANY_0001", "COMPANY_0093"]
        batch_label = "demo-three"
        limit = len(selected)
    elif args.ids:
        selected = [x.strip() for x in args.ids.split(",") if x.strip()]
        batch_label = "ids"
        limit = len(selected)
    else:
        if limit is None:
            limit = settings.resolve_batch_limit(len(all_ids))
            batch_label = settings.batch_size
        else:
            batch_label = str(limit)
        selected = all_ids[args.offset : args.offset + limit]

    missing = [cid for cid in selected if cid not in set(all_ids)]
    if missing:
        logger.error("Unknown company IDs: %s", missing)
        return 1

    if not selected:
        logger.error(
            "No companies selected (offset=%s limit=%s batch_size=%s total=%s)",
            args.offset,
            limit,
            settings.batch_size,
            len(all_ids),
        )
        return 1

    logger.info(
        "Batch start — provider=%s model=%s companies=%d batch_size=%s offset=%d",
        settings.llm_provider,
        settings.llm_model,
        len(selected),
        batch_label,
        args.offset,
    )

    agent = CreditAgent(settings)
    grouped: dict[str, list[dict]] = defaultdict(list)
    errors: list[dict] = []
    rows: list[dict] = []

    t0 = time.perf_counter()
    for i, company_id in enumerate(selected, start=1):
        profile = loader.get_company(company_id)
        started = time.perf_counter()
        result = agent.assess(profile)
        elapsed = time.perf_counter() - started

        if result.error:
            bucket = "Error"
            rec_label = "Error"
            errors.append({"company_id": company_id, "error": result.error})
        else:
            rec_label = result.recommendation.value  # Continue | Review | No-Go
            bucket = rec_label

        row = {
            "company_id": company_id,
            "recommendation": rec_label,
            "confidence": result.confidence.value if not result.error else None,
            "guardrail_fired": result.guardrail_fired,
            "guardrail_reason": result.guardrail_reason or None,
            "rationale": (result.rationale or "")[:300],
            "error": result.error,
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)
        grouped[bucket].append(row)

        logger.info(
            "[%d/%d] %s → %s (%.1fs)%s",
            i,
            len(selected),
            company_id,
            rec_label,
            elapsed,
            " [guardrail]" if result.guardrail_fired else "",
        )

    total_s = time.perf_counter() - t0

    # Stable display order
    order = [
        Recommendation.CONTINUE.value,
        Recommendation.REVIEW.value,
        Recommendation.NO_GO.value,
        "Error",
    ]

    ids_by_rec: dict[str, list[str]] = {
        label: [item["company_id"] for item in grouped.get(label, [])]
        for label in order
    }

    print("\n" + "=" * 64)
    print(f"BATCH SUMMARY — {len(selected)} companies in {total_s/60:.1f} min")
    print(f"Model: {settings.llm_provider} / {settings.llm_model}")
    print("=" * 64)

    print("\n>>> COMPANY IDs BY VERDICT <<<\n")
    for label in order:
        ids = ids_by_rec.get(label, [])
        if not ids and label == "Error":
            continue
        print(f"{label} ({len(ids)}):")
        if not ids:
            print("  (none)\n")
            continue
        # Print in compact wrapped lines of up to 6 IDs
        for start in range(0, len(ids), 6):
            chunk = ids[start : start + 6]
            print("  " + ", ".join(chunk))
        print()

    print("-" * 64)
    print("Counts:")
    for label in order:
        n = len(ids_by_rec.get(label, []))
        if n or label != "Error":
            pct = 100.0 * n / len(selected) if selected else 0
            print(f"  {label:10s}  {n:4d}  ({pct:5.1f}%)")
    print("-" * 64)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "batch_size": batch_label,
        "offset": args.offset,
        "limit": limit,
        "assessed": len(selected),
        "elapsed_seconds": round(total_s, 1),
        "counts": {label: len(ids_by_rec.get(label, [])) for label in order},
        # Easy lookup: which IDs landed in each bucket
        "ids_by_recommendation": {
            label: ids_by_rec.get(label, [])
            for label in order
            if label != "Error" or ids_by_rec.get(label)
        },
        "grouped": {label: grouped.get(label, []) for label in order},
        "rows": rows,
    }

    out_path = args.out
    if out_path is None:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = str(batch_label).replace("/", "-").replace(" ", "_")
        out_path = out_dir / f"batch_{safe_label}_{stamp}.json"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plain-text ID lists — one file per verdict for easy copy/paste
    ids_dir = out_path.with_name(out_path.stem + "_ids")
    ids_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"Batch ID summary — {len(selected)} companies",
        f"Model: {settings.llm_provider} / {settings.llm_model}",
        f"Generated: {payload['generated_at']}",
        "",
    ]
    for label in order:
        ids = ids_by_rec.get(label, [])
        if not ids and label == "Error":
            continue
        safe = label.replace("/", "-").replace(" ", "_")
        list_path = ids_dir / f"{safe}.txt"
        list_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        summary_lines.append(f"{label} ({len(ids)}):")
        summary_lines.extend(f"  {cid}" for cid in ids)
        summary_lines.append("")

    summary_path = ids_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nWrote detailed JSON  → {out_path}")
    print(f"Wrote ID lists       → {ids_dir}/")
    print(f"  Continue.txt / Review.txt / No-Go.txt / SUMMARY.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
