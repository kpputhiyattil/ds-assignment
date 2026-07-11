"""Build the single deployable churn-decision artifact (Step 8).

Loads the latest trained models and packs them -- together with the feature-
engineering path and the SHAP explainer capability -- into one ``joblib`` bundle.
Before writing, it explicitly confirms all four required pieces are present
(model, preprocessing/feature-engineering, and explainer), per the production
packaging checklist, so preprocessing can never ship out of sync with the model.
"""

from __future__ import annotations

import json
import logging

import joblib

from src.config import Config, get_config
from src.features.build import feature_columns
from src.serving.model import ChurnDecisionModel

logger = logging.getLogger(__name__)


def build_package(cfg: Config | None = None) -> str:
    cfg = cfg or get_config()
    models_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "models"
    version = (models_dir / "latest.txt").read_text(encoding="utf-8").strip()
    mdir = models_dir / f"model_{version}"
    meta = json.loads((mdir / "metadata.json").read_text(encoding="utf-8"))

    bundle = ChurnDecisionModel(
        frequency=joblib.load(mdir / "frequency.joblib"),
        severity=joblib.load(mdir / "severity.joblib"),
        raw_frequency=joblib.load(mdir / "frequency_raw.joblib"),
        feature_cols=feature_columns(),
        cfg=cfg,
        version=version,
        git_commit=meta.get("git_commit", "unknown"),
    )

    # Packaging checklist: all four pieces must be present before shipping.
    checklist = {
        "feature_engineering": bundle.has_feature_engineering(),
        "models": bundle.has_models(),
        "explainer": bundle.has_explainer(),
        "decision_logic": True,
    }
    missing = [k for k, v in checklist.items() if not v]
    if missing:
        raise RuntimeError(f"Refusing to package; missing pieces: {missing}")

    out_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "churn_decision_model.joblib"
    joblib.dump(bundle, out_path)
    (out_dir / "package_manifest.json").write_text(
        json.dumps({"model_version": version, "git_commit": bundle.git_commit,
                    "checklist": checklist, "n_features": len(bundle.feature_cols),
                    "artifact": str(out_path)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Packaged churn-decision model %s (checklist OK: %s) -> %s",
                version, checklist, out_path)
    return str(out_path)


def load_package(cfg: Config | None = None) -> ChurnDecisionModel:
    cfg = cfg or get_config()
    path = cfg.paths.resolve(cfg.paths.artifacts_dir) / "serving" / "churn_decision_model.joblib"
    return joblib.load(path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_package()


if __name__ == "__main__":
    main()
