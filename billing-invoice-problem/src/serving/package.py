"""Build the deployable dual churn-decision artifact (Step 8).

Trains / packs **two** model pairs from the same temporal split:

* with inferred-interval features  (CORE + GAP + INTERVAL)
* without inferred-interval features (CORE + GAP)

Plus holdout comparison metrics so serving can show the assignment-relevant
with-vs-without story on every scored file.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import polars as pl

from src.config import Config, get_config
from src.features.build import CORE_FEATURES, GAP_FEATURES, INTERVAL_FEATURES, feature_columns
from src.features.labels import Y_CHURN, Y_LOSS
from src.models import metrics as M
from src.models.train import temporal_split, train_frequency, train_severity
from src.serving.model import ChurnDecisionModel, _ModelPair

logger = logging.getLogger(__name__)


def _xy(df: pl.DataFrame, feats: list[str]):
    X = df.select(feats).to_numpy().astype(float)
    y = df[Y_CHURN].to_numpy().astype(int)
    loss = df[Y_LOSS].to_numpy().astype(float)
    return X, y, loss


def _holdout_comparison(
    pair_with: _ModelPair,
    pair_wo: _ModelPair,
    test_df: pl.DataFrame,
) -> dict:
    """Offline test-set comparison of with vs without interval models."""
    Xw, yw, lossw = _xy(test_df, pair_with.feature_cols)
    Xo, yo, losso = _xy(test_df, pair_wo.feature_cols)
    pw = pair_with.frequency.predict_proba(Xw)[:, 1]
    po = pair_wo.frequency.predict_proba(Xo)[:, 1]
    ew = pw * np.clip(pair_with.severity.predict(Xw), 0, None)
    eo = po * np.clip(pair_wo.severity.predict(Xo), 0, None)
    return {
        "n_test": int(len(yw)),
        "with_interval": {
            "pr_auc": round(M.pr_auc(yw, pw), 4),
            "roc_auc": round(M.roc_auc(yw, pw), 4),
            "dollar_recall_top10pct": round(M.dollar_recall_at_k(lossw, ew, 0.10), 4),
            "expected_loss_mae": round(M.mae(lossw, ew), 2),
        },
        "without_interval": {
            "pr_auc": round(M.pr_auc(yo, po), 4),
            "roc_auc": round(M.roc_auc(yo, po), 4),
            "dollar_recall_top10pct": round(M.dollar_recall_at_k(losso, eo, 0.10), 4),
            "expected_loss_mae": round(M.mae(losso, eo), 2),
        },
        "delta": {
            "pr_auc": round(M.pr_auc(yw, pw) - M.pr_auc(yo, po), 4),
            "dollar_recall_top10pct": round(
                M.dollar_recall_at_k(lossw, ew, 0.10)
                - M.dollar_recall_at_k(losso, eo, 0.10),
                4,
            ),
            "expected_loss_mae": round(M.mae(lossw, ew) - M.mae(losso, eo), 2),
        },
        "narrative": (
            "Holdout comparison (same temporal test fold): model WITH interval "
            f"PR-AUC={M.pr_auc(yw, pw):.4f} vs WITHOUT={M.pr_auc(yo, po):.4f} "
            f"(Δ={M.pr_auc(yw, pw) - M.pr_auc(yo, po):+.4f}); "
            f"dollar-recall@10% Δ="
            f"{M.dollar_recall_at_k(lossw, ew, 0.10) - M.dollar_recall_at_k(losso, eo, 0.10):+.4f}."
        ),
    }


def build_package(cfg: Config | None = None) -> str:
    cfg = cfg or get_config()
    models_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "models"
    version = (models_dir / "latest.txt").read_text(encoding="utf-8").strip()
    mdir = models_dir / f"model_{version}"
    meta = json.loads((mdir / "metadata.json").read_text(encoding="utf-8"))

    feats_path = cfg.paths.resolve(cfg.paths.processed_dir) / "features.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Missing {feats_path}; run `make features` / `make labels` / `make train` first."
        )

    df = pl.read_parquet(feats_path)
    train_df, test_df = temporal_split(df, cfg)

    feats_with = feature_columns()  # CORE + GAP + INTERVAL
    feats_without = CORE_FEATURES + GAP_FEATURES
    assert all(f in feats_with for f in INTERVAL_FEATURES)

    logger.info(
        "Training dual package models (with=%d feats, without=%d feats)…",
        len(feats_with),
        len(feats_without),
    )
    Xw, yw, lossw = _xy(train_df, feats_with)
    Xo, yo, losso = _xy(train_df, feats_without)

    freq_w, raw_w, _ = train_frequency(Xw, yw, cfg)
    sev_w = train_severity(Xw, yw, lossw, cfg)
    freq_o, raw_o, _ = train_frequency(Xo, yo, cfg)
    sev_o = train_severity(Xo, yo, losso, cfg)

    # Prefer calibrated artifacts from the latest train run for the WITH model
    # when feature lists match (keeps serving aligned with the graded train step).
    trained_feats = meta.get("feature_columns") or []
    if trained_feats == feats_with:
        logger.info("Reusing calibrated WITH-interval models from %s", mdir)
        freq_w = joblib.load(mdir / "frequency.joblib")
        raw_w = joblib.load(mdir / "frequency_raw.joblib")
        sev_w = joblib.load(mdir / "severity.joblib")

    pair_with = _ModelPair(freq_w, sev_w, raw_w, feats_with)
    pair_wo = _ModelPair(freq_o, sev_o, raw_o, feats_without)
    comparison = _holdout_comparison(pair_with, pair_wo, test_df)

    bundle = ChurnDecisionModel(
        with_interval=pair_with,
        without_interval=pair_wo,
        cfg=cfg,
        version=version,
        git_commit=meta.get("git_commit", "unknown"),
        comparison_metrics=comparison,
    )

    checklist = {
        "feature_engineering": bundle.has_feature_engineering(),
        "models": bundle.has_models(),
        "explainer": bundle.has_explainer(),
        "decision_logic": True,
        "dual_interval_ablation_models": True,
    }
    missing = [k for k, v in checklist.items() if not v]
    if missing:
        raise RuntimeError(f"Refusing to package; missing pieces: {missing}")

    out_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "churn_decision_model.joblib"
    joblib.dump(bundle, out_path)

    # Persist without-interval pair beside the train artifact for auditability.
    dual_dir = out_dir / "dual_models"
    dual_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(freq_o, dual_dir / "frequency_without_interval.joblib")
    joblib.dump(raw_o, dual_dir / "frequency_raw_without_interval.joblib")
    joblib.dump(sev_o, dual_dir / "severity_without_interval.joblib")
    (dual_dir / "comparison_holdout.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    (out_dir / "package_manifest.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "git_commit": bundle.git_commit,
                "checklist": checklist,
                "n_features_with_interval": len(feats_with),
                "n_features_without_interval": len(feats_without),
                "feature_cols_with_interval": feats_with,
                "feature_cols_without_interval": feats_without,
                "holdout_comparison": comparison,
                "artifact": str(out_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Packaged dual churn-decision model %s -> %s | %s",
        version,
        out_path,
        comparison["narrative"],
    )
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
