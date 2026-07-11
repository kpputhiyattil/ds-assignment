"""
export_landlord_scorer.py — Pack FE pipeline + best model + SHAP into LandlordScorer.

Loads the recommended model (or --model), feature matrix, optional OOF preds /
SHAP importance, and writes:

  data/artifacts/landlord_scorer.joblib
  data/artifacts/landlord_scorer_meta.json

The artifact includes:
  - training-aligned feature-engineering pipeline (transform / score_end_to_end)
  - fitted regressor
  - TreeSHAP explainability
  - percentile quality bands + confidence rules

Usage:
    python scripts/export_landlord_scorer.py
    python scripts/export_landlord_scorer.py --model catboost
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
from src.features.transforms import get_feature_cols
from src.models.scorer import PACKAGE_COMPONENTS, PIPELINE_STEPS, build_scorer_from_artifacts
from src.utils.reproducibility import repo_relpath, set_seed

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
            logger.info("Using recommended model: %s", best)
            return str(best)
    return "catboost"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Export LandlordScorer artifact")
    parser.add_argument("--model", default=None, help="Model type (default: recommendation)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output joblib path (default: data/artifacts/landlord_scorer.joblib)",
    )
    args = parser.parse_args(argv)

    seed = set_seed(int(cfg.get("seed", 42)))
    model_type = _resolve_model_type(args.model)

    processed = Path(cfg["paths"]["processed_dir"])
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    matrix_path = processed / "landlord_feature_matrix.parquet"
    model_path = artifacts / f"landlord_model_{model_type}.joblib"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {matrix_path}. Run scripts/run_landlord_models.py first."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    matrix = pd.read_parquet(matrix_path)
    model = joblib.load(model_path)
    numeric_cols, categorical_cols = get_feature_cols(matrix)
    feature_set = "with_portfolio"

    # Optional metrics / importance / OOF reference
    metrics: dict = {}
    eval_path = reports / "landlord_model_eval_summary.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as fp:
            eval_summary = json.load(fp)
        feature_set = eval_summary.get("feature_set", feature_set)
        metrics = (eval_summary.get("models") or {}).get(model_type, {})

    importance = None
    shap_imp = reports / "shap_global_importance.csv"
    model_imp = reports / f"feature_importance_{model_type}.csv"
    if shap_imp.exists():
        importance = pd.read_csv(shap_imp)
    elif model_imp.exists():
        importance = pd.read_csv(model_imp)

    reference_scores = None
    oof_path = artifacts / "landlord_oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        if "model_type" in oof.columns:
            oof = oof[oof["model_type"] == model_type]
        if "oof_pred" in oof.columns and len(oof):
            reference_scores = oof["oof_pred"].to_numpy(dtype=float)
            logger.info("Using %d OOF predictions as reference scores", len(reference_scores))

    rec = {}
    rec_path = reports / "model_comparison_recommendation.json"
    if rec_path.exists():
        with open(rec_path, encoding="utf-8") as fp:
            rec = json.load(fp)

    version = (
        f"{model_type}-seed{seed}-"
        f"{pd.Timestamp.utcnow().strftime('%Y%m%d')}"
    )

    scorer = build_scorer_from_artifacts(
        model=model,
        model_type=model_type,
        matrix=matrix,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cfg=cfg,
        feature_set=feature_set,
        metrics={
            "oof": metrics.get("oof_metrics") or metrics.get("mean_metrics") or {},
            "recommendation_reason": rec.get("reason"),
            "best_model": rec.get("best_model", model_type),
        },
        global_importance=importance,
        reference_scores=reference_scores,
        version=version,
    )

    out_path = Path(args.out) if args.out else artifacts / "landlord_scorer.joblib"
    scorer.save(out_path)

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_payload = {
        **scorer.metadata_dict(),
        "artifact_path": repo_relpath(out_path),
        "source_model_path": repo_relpath(model_path),
        "matrix_path": repo_relpath(matrix_path),
        "pipeline_packed": True,
        "pipeline_steps": list(PIPELINE_STEPS),
        "package_components": list(PACKAGE_COMPONENTS),
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2, default=str)

    # Smoke: feature-matrix path
    sample = matrix.head(3)
    scored = scorer.score_with_explanation(sample, top_k=3)
    manifest = scorer.package_manifest()

    print("\n" + "=" * 72)
    print("LANDLORD SCORER ARTIFACT EXPORTED")
    print("=" * 72)
    print(f"Model:     {model_type}")
    print(f"Version:   {scorer.meta.version}")
    print(f"Packaged:  {', '.join(PACKAGE_COMPONENTS)}")
    for name in PACKAGE_COMPONENTS:
        block = manifest.get(name, {})
        print(f"  [{name}] included={block.get('included')}")
    print(f"Pipeline:  {scorer.meta.pipeline_name}")
    print(f"  Steps:   {' -> '.join(scorer.meta.pipeline_steps)}")
    print(f"Artifact:  {repo_relpath(out_path)}")
    print(f"Meta:      {repo_relpath(meta_path)}")
    print(f"Reference: {len(scorer.reference_scores)} scores")
    print("-" * 72)
    print("Smoke score_with_explanation (first 3):")
    for row in scored:
        print(f"  {row.get('LandLordID')}: score={row['PredictedScore']:.4f} "
              f"band={row['QualityBand']} conf={row['ConfidenceLevel']}")
        print(f"    {row.get('ExplanationSummary', '')[:160]}")
    print("=" * 72)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
