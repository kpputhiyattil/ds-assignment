"""
run_shap_explainability.py — SHAP global plots + landlord case studies.

Loads the recommended (or specified) landlord model, computes TreeSHAP on a
sample of the feature matrix, writes global plots, dependence plots, and
good / bad / neutral case studies with local waterfall charts.

Usage (from project root, venv active):
    python scripts/run_shap_explainability.py
    python scripts/run_shap_explainability.py --model catboost --max-samples 1500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import cfg
from src.explainability.shap_utils import (
    build_case_studies,
    compute_shap_explanation,
    mean_abs_shap_table,
    plot_beeswarm,
    plot_dependence,
    plot_global_importance,
    prepare_model_frame,
    select_case_study_ids,
    write_explainability_report,
)
from src.features.transforms import get_feature_cols

logger = logging.getLogger(__name__)


def _resolve_model_type(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    rec_path = Path(cfg["paths"]["reports_dir"]) / "model_comparison_recommendation.json"
    if rec_path.exists():
        with open(rec_path, encoding="utf-8") as fp:
            rec = json.load(fp)
        best = rec.get("best_model")
        if best:
            logger.info("Using recommended model from %s: %s", rec_path, best)
            return str(best)
    return "catboost"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="SHAP explainability + case studies")
    parser.add_argument("--model", default=None, help="Model type (default: recommendation or catboost)")
    parser.add_argument("--max-samples", type=int, default=2000, help="Rows for global SHAP")
    parser.add_argument("--n-good", type=int, default=2)
    parser.add_argument("--n-bad", type=int, default=2)
    parser.add_argument("--n-mid", type=int, default=1)
    parser.add_argument("--min-tenants", type=int, default=5)
    parser.add_argument("--n-dependence", type=int, default=3, help="Top features for dependence plots")
    args = parser.parse_args(argv)

    model_type = _resolve_model_type(args.model)
    processed = Path(cfg["paths"]["processed_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    figures = reports / "figures" / "shap"
    figures.mkdir(parents=True, exist_ok=True)

    matrix_path = processed / "landlord_feature_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {matrix_path}. Run scripts/run_landlord_models.py first."
        )
    model_path = artifacts / f"landlord_model_{model_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    matrix = pd.read_parquet(matrix_path)
    model = joblib.load(model_path)
    numeric_cols, categorical_cols = get_feature_cols(matrix)
    X = prepare_model_frame(matrix, numeric_cols, categorical_cols, model_type=model_type)

    # OOF predictions for case study context
    oof_path = artifacts / "landlord_oof_predictions.parquet"
    scores = matrix[["LandLordID", "AdjustedScore", "TenantCount"]].copy()
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        oof_m = oof[oof["model_type"] == model_type] if "model_type" in oof.columns else oof
        if "oof_pred" in oof_m.columns:
            scores = scores.merge(
                oof_m[["LandLordID", "oof_pred"]].drop_duplicates("LandLordID"),
                on="LandLordID",
                how="left",
            )

    case_rows = select_case_study_ids(
        scores,
        n_good=args.n_good,
        n_bad=args.n_bad,
        n_mid=args.n_mid,
        min_tenants=args.min_tenants,
    )
    force_ids = set(case_rows["LandLordID"].astype(str))
    force_index = matrix.index[matrix["LandLordID"].astype(str).isin(force_ids)].tolist()

    explanation = compute_shap_explanation(
        model,
        X,
        model_type=model_type,
        max_samples=args.max_samples,
        seed=int(cfg.get("seed", 42)),
        force_index=force_index,
    )

    # Global plots
    fig_paths: dict[str, str] = {}
    fig_paths["importance"] = str(
        plot_global_importance(
            explanation,
            figures / "shap_global_importance.png",
            title=f"SHAP global importance ({model_type})",
        )
    )
    fig_paths["beeswarm"] = str(
        plot_beeswarm(
            explanation,
            figures / "shap_beeswarm.png",
            title=f"SHAP beeswarm ({model_type})",
        )
    )

    importance = mean_abs_shap_table(explanation, top_n=20)
    importance.to_csv(reports / "shap_global_importance.csv", index=False)

    for feat in importance["feature"].head(args.n_dependence).tolist():
        key = f"dependence_{feat}"
        safe = feat.replace("/", "_").replace(" ", "_")
        try:
            fig_paths[key] = str(
                plot_dependence(
                    explanation,
                    feat,
                    figures / f"shap_dependence_{safe}.png",
                    title=f"SHAP dependence: {feat}",
                )
            )
        except Exception as exc:
            logger.warning("Dependence plot failed for %s: %s", feat, exc)

    # Case studies (with waterfalls)
    studies = build_case_studies(
        matrix,
        explanation,
        case_rows,
        figures_dir=figures / "case_studies",
    )
    studies_path = reports / "shap_case_studies.json"
    with open(studies_path, "w", encoding="utf-8") as fp:
        json.dump(studies, fp, indent=2, default=str)

    report_path = write_explainability_report(
        model_type=model_type,
        global_importance=importance,
        case_studies=studies,
        figure_paths=fig_paths,
        out_path=reports / "shap_explainability_report.md",
        n_shap_rows=len(explanation.values),
        n_landlords=len(matrix),
    )

    print("\n" + "=" * 72)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 72)
    print(f"Model:     {model_type}")
    print(f"SHAP rows: {len(explanation.values)}")
    print(f"Report:    {report_path}")
    print(f"Cases:     {studies_path}")
    print(f"Figures:   {figures}")
    print("Top features (mean |SHAP|):")
    for _, row in importance.head(8).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.5f}")
    print("Case studies:")
    for cs in studies:
        print(f"  [{cs['band']}] {cs['LandLordID']} score={cs['AdjustedScore']:.4f}")
    print("=" * 72)

    return report_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
