"""
compare_landlord_models.py — compare trained models, recommend best.

Reads the comparison CSV and eval summary already produced by
``run_landlord_models.py`` and adds:
  - model_comparison_recommendation.json  (best-model pick + rationale)
  - model_comparison_report.md            (human-readable markdown)

If the comparison CSV does not exist yet, falls back to training
all models from scratch (same as run_landlord_models).

Usage (from project root, venv active):
    python scripts/compare_landlord_models.py
    python scripts/compare_landlord_models.py --retrain   # force retrain all models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import cfg
from src.models.evaluate import (
    render_model_comparison_report,
    suggest_best_model,
)
from src.utils.reproducibility import repo_relpath, set_seed

logger = logging.getLogger(__name__)


def _train_and_compare() -> tuple[pd.DataFrame, str, int]:
    """Full training fallback when cached artifacts are missing."""
    from src.data.ingestion import build_model_base, load_companies, load_landlords
    from src.features.transforms import build_feature_matrix, get_feature_cols
    from src.models.company_baseline import build_landlord_adjusted_scores
    from src.models.evaluate import compare_models
    from src.models.train import ALL_MODEL_TYPES, train_all
    from src.targets.construction import build_targets

    processed = Path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    feature_set = "with_portfolio"
    fs_cfg = cfg.get("landlord_model", {}).get("feature_sets", {})
    if fs_cfg.get("with_portfolio") is False:
        feature_set = "landlord_only"

    matrix_path = processed / "landlord_feature_matrix.parquet"
    if matrix_path.exists():
        matrix = pd.read_parquet(matrix_path)
    else:
        landlords = load_landlords()
        companies = load_companies()
        bridge, model_base = build_model_base(landlords, companies)
        targets_path = processed / "company_targets.parquet"
        if targets_path.exists():
            companies_scored = pd.read_parquet(targets_path)
        else:
            companies_scored = build_targets(companies, cfg=cfg)
            companies_scored.to_parquet(targets_path, index=False)
        target_cols = [
            c for c in ("CompanyID", "CompanyIsActive", "SuccessScore", "SuccessScoreRank")
            if c in companies_scored.columns
        ]
        model_base = model_base.merge(
            companies_scored[target_cols], on="CompanyID", how="left"
        )
        model_base_oof, landlord_scores = build_landlord_adjusted_scores(
            model_base, bridge, cfg=cfg
        )
        matrix = build_feature_matrix(
            landlords, model_base_oof, landlord_scores, cfg=cfg, feature_set=feature_set
        )
        matrix.to_parquet(matrix_path, index=False)

    numeric_cols, categorical_cols = get_feature_cols(matrix, feature_set=feature_set)
    results = train_all(
        matrix,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cfg=cfg,
        model_types=list(ALL_MODEL_TYPES),
    )
    comparison = compare_models(results)
    return comparison, feature_set, len(matrix)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compare landlord regression models")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain all models instead of reading cached comparison",
    )
    args = parser.parse_args(argv)

    seed = set_seed(int(cfg.get("seed", 42)))
    reports = Path(cfg["paths"]["reports_dir"])
    processed = Path(cfg["paths"]["processed_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    logger.info("Reproducibility seed=%d", seed)

    csv_path = reports / "landlord_model_comparison.csv"
    feature_set = "with_portfolio"
    fs_cfg = cfg.get("landlord_model", {}).get("feature_sets", {})
    if fs_cfg.get("with_portfolio") is False:
        feature_set = "landlord_only"

    if csv_path.exists() and not args.retrain:
        logger.info("Reading existing comparison from %s (skip retraining)", csv_path)
        comparison = pd.read_csv(csv_path)
        matrix_path = processed / "landlord_feature_matrix.parquet"
        if matrix_path.exists():
            n_landlords = len(pd.read_parquet(matrix_path, columns=["LandLordID"]))
        elif "n_landlords" in comparison.columns and len(comparison):
            n_landlords = int(comparison["n_landlords"].iloc[0])
        else:
            n_landlords = None
    else:
        if args.retrain:
            logger.info("--retrain flag set, training all models …")
        else:
            logger.info("No cached comparison found, training all models …")
        comparison, feature_set, n_landlords = _train_and_compare()
        comparison.to_csv(csv_path, index=False)

    suggestion = suggest_best_model(comparison)

    n_folds = int(comparison["n_folds"].iloc[0]) if len(comparison) and "n_folds" in comparison.columns else None
    md = render_model_comparison_report(
        comparison,
        suggestion,
        n_landlords=n_landlords,
        n_folds=n_folds,
        feature_set=feature_set,
        extra_notes=[
            f"Comparison CSV: `{repo_relpath(csv_path)}`",
            "Primary selection rule: lowest GroupKFold MAE, then highest Spearman.",
            "CatBoost uses native categoricals; other models use ordinal-encoded categories.",
            f"Config seed: {seed}.",
        ],
    )
    report_path = reports / "model_comparison_report.md"
    report_path.write_text(md, encoding="utf-8")

    rec_path = reports / "model_comparison_recommendation.json"
    rec_payload = {
        "best_model": suggestion["best_model"],
        "reason": suggestion["reason"],
        "winners_by_metric": suggestion["winners_by_metric"],
        "best_row": suggestion.get("best_row"),
        "comparison": comparison.to_dict(orient="records"),
        "seed": seed,
        "report_path": repo_relpath(report_path),
        "csv_path": repo_relpath(csv_path),
    }
    with open(rec_path, "w", encoding="utf-8") as fp:
        json.dump(rec_payload, fp, indent=2, default=str)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 72)
    print(comparison.to_string(index=False))
    print("-" * 72)
    print(f"BEST MODEL: {suggestion['best_model']}")
    print(suggestion["reason"])
    print("-" * 72)
    print(f"Report:  {report_path}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {rec_path}")
    print("=" * 72)

    return report_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
