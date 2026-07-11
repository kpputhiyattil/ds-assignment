"""
run_landlord_models.py — end-to-end landlord quality modeling on real data.

Pipeline (matches project document stages 4–7)
---------------------------------------------
1. Load Landlords / Companies + build bridge / model_base
2. Attach company success targets (CompanyIsActive, SuccessScore)
3. Company baseline OOF predictions (GroupKFold by LandLordID) → residuals
4. Aggregate + Empirical-Bayes shrink → AdjustedScore per landlord
5. Build landlord feature matrix
6. Train CatBoost / XGBoost / LightGBM / RandomForest with GroupKFold
7. Evaluate (MAE, RMSE, R², Spearman) and write artifacts

Usage (from project root, venv active):
    python scripts/run_landlord_models.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
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
    lift_table,
    quality_band_metrics,
    regression_metrics,
)
from src.models.train import ALL_MODEL_TYPES, train_all
from src.targets.construction import build_targets

logger = logging.getLogger(__name__)


def _attach_targets(model_base: pd.DataFrame, companies_scored: pd.DataFrame) -> pd.DataFrame:
    """Merge company target columns onto the landlord–company model base."""
    target_cols = [
        c for c in ("CompanyID", "CompanyIsActive", "SuccessScore", "SuccessScoreRank")
        if c in companies_scored.columns
    ]
    # Drop any pre-existing target cols from the companies side of the join
    drop_existing = [c for c in target_cols if c != "CompanyID" and c in model_base.columns]
    base = model_base.drop(columns=drop_existing, errors="ignore")
    return base.merge(companies_scored[target_cols], on="CompanyID", how="left")


def main() -> Path:
    processed = Path(cfg["paths"]["processed_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + bridge ─────────────────────────────────────────────────────
    landlords = load_landlords()
    companies = load_companies()
    bridge, model_base = build_model_base(landlords, companies)

    # ── 2. Company targets ───────────────────────────────────────────────────
    targets_path = processed / "company_targets.parquet"
    if targets_path.exists():
        logger.info("Loading existing company targets from %s", targets_path)
        companies_scored = pd.read_parquet(targets_path)
    else:
        logger.info("Building company targets …")
        companies_scored = build_targets(companies, cfg=cfg)
        companies_scored.to_parquet(targets_path, index=False)

    model_base = _attach_targets(model_base, companies_scored)

    # ── 3–4. Company baseline OOF + landlord AdjustedScore ───────────────────
    logger.info("Computing GroupKFold OOF company baseline + landlord scores …")
    model_base_oof, landlord_scores = build_landlord_adjusted_scores(
        model_base, bridge, cfg=cfg
    )
    model_base_oof.to_parquet(processed / "model_base_oof.parquet", index=False)
    landlord_scores.to_parquet(processed / "landlord_adjusted_scores.parquet", index=False)

    # ── 5. Feature matrix ────────────────────────────────────────────────────
    feature_set = "with_portfolio"
    fs_cfg = cfg.get("landlord_model", {}).get("feature_sets", {})
    if fs_cfg.get("with_portfolio") is False:
        feature_set = "landlord_only"

    matrix = build_feature_matrix(
        landlords, model_base_oof, landlord_scores, cfg=cfg, feature_set=feature_set
    )
    matrix.to_parquet(processed / "landlord_feature_matrix.parquet", index=False)
    numeric_cols, categorical_cols = get_feature_cols(matrix, feature_set=feature_set)

    logger.info(
        "Feature matrix: %d landlords, %d numeric, %d categorical (set=%s)",
        len(matrix), len(numeric_cols), len(categorical_cols), feature_set,
    )

    # ── 6. Train all models (GroupKFold) ─────────────────────────────────────
    logger.info("Training models: %s", ALL_MODEL_TYPES)
    results = train_all(
        matrix,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cfg=cfg,
        model_types=list(ALL_MODEL_TYPES),
    )

    # ── 7. Evaluation + artifacts ────────────────────────────────────────────
    comparison = compare_models(results)
    comparison_path = reports / "landlord_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    eval_payload: dict = {
        "feature_set": feature_set,
        "n_landlords": int(len(matrix)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "models": {},
    }

    eval_cfg = cfg.get("evaluation", {})
    top_pct = float(cfg.get("landlord_score", {}).get("top_percentile", 70)) / 100.0
    bot_pct = float(cfg.get("landlord_score", {}).get("bottom_percentile", 30)) / 100.0

    for model_type, cv_result in results.items():
        # Persist final model
        model_path = artifacts / f"landlord_model_{model_type}.joblib"
        joblib.dump(cv_result.final_model, model_path)

        # Persist feature importance
        imp_path = reports / f"feature_importance_{model_type}.csv"
        cv_result.feature_importance.to_csv(imp_path, index=False)

        oof_metrics = regression_metrics(cv_result.oof_true, cv_result.oof_predictions, prefix="oof_")
        bands = quality_band_metrics(
            cv_result.oof_true,
            cv_result.oof_predictions,
            top_percentile=top_pct,
            bottom_percentile=bot_pct,
        )
        lift = lift_table(cv_result.oof_true, cv_result.oof_predictions, n_bins=10)
        lift.to_csv(reports / f"lift_table_{model_type}.csv", index=False)

        fold_rows = [
            {
                "fold": f.fold,
                "mae": f.mae,
                "rmse": f.rmse,
                "r2": f.r2,
                "spearman": f.spearman,
                "n_val": f.n_val,
            }
            for f in cv_result.fold_results
        ]
        pd.DataFrame(fold_rows).to_csv(reports / f"cv_folds_{model_type}.csv", index=False)

        eval_payload["models"][model_type] = {
            "mean_metrics": cv_result.mean_metrics,
            "std_metrics": cv_result.std_metrics,
            "oof_metrics": oof_metrics,
            "quality_bands": bands,
            "model_path": str(model_path),
            "importance_path": str(imp_path),
        }

        logger.info(
            "[%s] OOF MAE=%.4f RMSE=%.4f R2=%.4f Spearman=%.4f",
            model_type,
            oof_metrics["oof_mae"],
            oof_metrics["oof_rmse"],
            oof_metrics["oof_r2"],
            oof_metrics["oof_spearman"],
        )

    summary_path = reports / "landlord_model_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(eval_payload, fp, indent=2, default=str)

    # Console summary
    print("\n" + "=" * 72)
    print("LANDLORD MODEL TRAINING COMPLETE (GroupKFold)")
    print("=" * 72)
    print(comparison.to_string(index=False))
    print("-" * 72)
    best = comparison.iloc[0]["model_type"] if len(comparison) else "n/a"
    print(f"Best by MAE: {best}")
    print(f"Comparison:  {comparison_path}")
    print(f"Summary:     {summary_path}")
    print(f"Scores:      {processed / 'landlord_adjusted_scores.parquet'}")
    print(f"Matrix:      {processed / 'landlord_feature_matrix.parquet'}")
    print("=" * 72)

    return summary_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
