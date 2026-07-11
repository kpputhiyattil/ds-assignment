"""
run_company_targets.py — build company success targets on actual Companies data.

Usage (from project root, venv active):
    python scripts/run_company_targets.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import cfg
from src.data.ingestion import load_companies
from src.targets.construction import build_targets, sensitivity_analysis
from src.utils.reproducibility import repo_relpath, set_seed


def main() -> Path:
    """Load Companies.parquet, build targets, write processed artifacts."""
    set_seed(int(cfg.get("seed", 42)))
    companies = load_companies()
    scored = build_targets(companies, cfg=cfg)
    sens = sensitivity_analysis(scored)

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "company_targets.parquet"
    scored.to_parquet(out_path, index=False)

    off_diag = sens.values[~np.eye(len(sens), dtype=bool)]
    summary = {
        "n_companies": int(len(scored)),
        "n_binary_labeled": int(scored["CompanyIsActive"].notna().sum()),
        "n_active": int((scored["CompanyIsActive"] == 1).sum()),
        "n_inactive": int((scored["CompanyIsActive"] == 0).sum()),
        "success_score": {
            "mean": float(scored["SuccessScore"].mean()),
            "median": float(scored["SuccessScore"].median()),
            "std": float(scored["SuccessScore"].std()),
            "min": float(scored["SuccessScore"].min()),
            "max": float(scored["SuccessScore"].max()),
            "n_nan": int(scored["SuccessScore"].isna().sum()),
        },
        "sensitivity_spearman_min": float(off_diag.min()) if len(off_diag) else 1.0,
        "output_path": repo_relpath(out_path),
        "seed": int(cfg.get("seed", 42)),
    }
    summary_path = out_dir / "company_targets_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 60)
    print("COMPANY TARGET CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Wrote:    {out_path}")
    print(f"Summary:  {summary_path}")
    print(
        f"Labeled:  {summary['n_binary_labeled']} "
        f"({summary['n_active']} active / {summary['n_inactive']} inactive)"
    )
    print(
        f"SuccessScore mean={summary['success_score']['mean']:.3f} "
        f"median={summary['success_score']['median']:.3f}"
    )
    print(f"Sensitivity min Spearman={summary['sensitivity_spearman_min']:.3f}")
    print("=" * 60)

    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
