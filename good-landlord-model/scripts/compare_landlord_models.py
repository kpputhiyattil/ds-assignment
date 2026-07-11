"""
compare_landlord_models.py — train landlord models, compare metrics, recommend best.

Uses existing processed feature matrix when available; otherwise builds the
pipeline (targets → OOF baseline → AdjustedScore → features).

Outputs
-------
  reports/landlord_model_comparison.csv
  reports/model_comparison_report.md
  reports/model_comparison_recommendation.json

Usage (from project root, venv active):
    python scripts/compare_landlord_models.py
    python scripts/compare_landlord_models.py --rebuild   # force rebuild matrix
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
from src.data.ingestion import build_model_base, load_companies, load_landlords
from src.features.transforms import build_feature_matrix, get_feature_cols
from src.models.company_baseline import build_landlord_adjusted_scores
from src.models.evaluate import (
    compare_models,
    render_model_comparison_report,
    suggest_best_model,
)
from src.models.train import ALL_MODEL_TYPES, train_all
from src.targets.construction import build_targets
from src.utils.reproducibility import repo_relpath, set_seed

logger = logging.getLogger(__name__)


def _attach_targets(model_base: pd.DataFrame, companies_scored: pd.DataFrame) -> pd.DataFrame:
    target_cols = [
        c for c in ("CompanyID", "CompanyIsActive", "SuccessScore", "SuccessScoreRank")
        if c in companies_scored.columns
    ]
    drop_existing = [c for c in target_cols if c != "CompanyID" and c in model_base.columns]
    base = model_base.drop(columns=drop_existing, errors="ignore")
    return base.merge(companies_scored[target_cols], on="CompanyID", how="left")


def _load_or_build_matrix(*, rebuild: bool) -> tuple[pd.DataFrame, str]:
    processed = Path(cfg["paths"]["processed_dir"])
    matrix_path = processed / "landlord_feature_matrix.parquet"
    feature_set = "with_portfolio"
    fs_cfg = cfg.get("landlord_model", {}).get("feature_sets", {})
    if fs_cfg.get("with_portfolio") is False:
        feature_set = "landlord_only"

    if matrix_path.exists() and not rebuild:
        logger.info("Loading feature matrix from %s", matrix_path)
        return pd.read_parquet(matrix_path), feature_set

    logger.info("Building feature matrix from raw data …")
    landlords = load_landlords()
    companies = load_companies()
    bridge, model_base = build_model_base(landlords, companies)

    targets_path = processed / "company_targets.parquet"
    if targets_path.exists() and not rebuild:
        companies_scored = pd.read_parquet(targets_path)
    else:
        companies_scored = build_targets(companies, cfg=cfg)
        processed.mkdir(parents=True, exist_ok=True)
        companies_scored.to_parquet(targets_path, index=False)

    model_base = _attach_targets(model_base, companies_scored)
    model_base_oof, landlord_scores = build_landlord_adjusted_scores(
        model_base, bridge, cfg=cfg
    )
    matrix = build_feature_matrix(
        landlords, model_base_oof, landlord_scores, cfg=cfg, feature_set=feature_set
    )
    processed.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(matrix_path, index=False)
    landlord_scores.to_parquet(processed / "landlord_adjusted_scores.parquet", index=False)
    return matrix, feature_set


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compare landlord regression models")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild targets / OOF scores / feature matrix even if cached",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_MODEL_TYPES),
        choices=list(ALL_MODEL_TYPES),
        help="Subset of models to train (default: all)",
    )
    args = parser.parse_args(argv)

    seed = set_seed(int(cfg.get("seed", 42)))
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    logger.info("Reproducibility seed=%d", seed)

    matrix, feature_set = _load_or_build_matrix(rebuild=args.rebuild)
    numeric_cols, categorical_cols = get_feature_cols(matrix, feature_set=feature_set)

    logger.info(
        "Training %s on %d landlords (%d num + %d cat features)",
        args.models, len(matrix), len(numeric_cols), len(categorical_cols),
    )
    results = train_all(
        matrix,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cfg=cfg,
        model_types=args.models,
    )

    comparison = compare_models(results)
    suggestion = suggest_best_model(comparison)

    csv_path = reports / "landlord_model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    n_folds = int(comparison["n_folds"].iloc[0]) if len(comparison) else None
    md = render_model_comparison_report(
        comparison,
        suggestion,
        n_landlords=len(matrix),
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
