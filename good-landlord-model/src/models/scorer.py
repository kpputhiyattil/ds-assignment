"""
scorer.py — Single serveable artifact packing:

  1. Feature engineering  (raw / IDs → landlord + portfolio features)
  2. Preprocessing        (schema align, cat fill, optional impute/encode)
  3. Model                (fitted regressor)
  4. SHAP                 (TreeSHAP local drivers)

``LandlordScorer`` load once, then:

  scorer.transform(upload_df)                 # feature engineering
  scorer.preprocess(features_df)              # model-ready matrix
  scorer.predict(features_df)
  scorer.explain(features_df)                 # SHAP
  scorer.score_with_explanation(features_df)
  scorer.score_end_to_end(upload_df)          # FE -> preprocess -> predict -> SHAP

Public API
----------
LandlordScorer.load(path) / .save(path)
build_scorer_from_artifacts(...)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.explainability.shap_utils import (
    local_top_drivers,
    prepare_model_frame,
)
from src.explainability.feature_glossary import (
    driver_phrase,
    enrich_driver,
    glossary_as_records,
)
from src.explainability.narrative import build_interpretation
from src.models.train import _CatEncoder, _NumImputer

logger = logging.getLogger(__name__)

# Four pillars packed into every LandlordScorer artifact
PACKAGE_COMPONENTS: tuple[str, ...] = (
    "model",
    "preprocessing",
    "feature_engineering",
    "shap",
)

# Canonical serving pipeline packed with the artifact
PIPELINE_STEPS: list[str] = [
    # --- feature engineering ---
    "detect_input_kind",
    "resolve_landlords",
    "build_model_base",
    "attach_company_targets",
    "landlord_features",
    "portfolio_features",
    # --- preprocessing ---
    "preprocess",
    # --- model ---
    "predict",
    # --- shap ---
    "shap_explain",
]


@dataclass
class ScorerMeta:
    """Serializable metadata stored alongside the model."""

    model_type: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    feature_set: str = "with_portfolio"
    seed: int = 42
    version: str = ""
    top_percentile: float = 70.0
    bottom_percentile: float = 30.0
    confidence_high_threshold: int = 10
    confidence_medium_threshold: int = 5
    n_reference: int = 0
    created_at: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    global_importance: list[dict[str, Any]] = field(default_factory=list)
    # FE / serving pipeline descriptor (packed with artifact)
    pipeline_steps: list[str] = field(default_factory=lambda: list(PIPELINE_STEPS))
    pipeline_name: str = "landlord_fe_v1"
    raw_landlord_hints: list[str] = field(
        default_factory=lambda: [
            "LandLordID",
            "AllCompanyID",
            "YearFounded",
            "Area",
            "TotalPopulationAround",
            "ActiveCompanies",
            "PreferredIndustry",
            "OriginCity",
            "OriginCountry",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LandlordScorer:
    """
    Packaged artifact: Feature engineering + Preprocessing + Model + SHAP.

    Attributes
    ----------
    model : fitted regressor (CatBoost / XGB / LGBM / RF)
    meta  : ScorerMeta (schema, thresholds, pipeline, version)
    reference_scores : 1-d array used for PercentileRank
    num_imputer / cat_encoder : fitted preprocessors (non-CatBoost);
        CatBoost uses native categoricals via ``prepare_model_frame``
    """

    def __init__(
        self,
        model: Any,
        meta: ScorerMeta,
        *,
        reference_scores: np.ndarray | None = None,
        num_imputer: _NumImputer | None = None,
        cat_encoder: _CatEncoder | None = None,
    ) -> None:
        self.model = model
        self.meta = meta
        self.reference_scores = (
            np.asarray(reference_scores, dtype=float)
            if reference_scores is not None
            else np.array([], dtype=float)
        )
        self.num_imputer = num_imputer
        self.cat_encoder = cat_encoder
        if not getattr(self.meta, "pipeline_steps", None):
            self.meta.pipeline_steps = list(PIPELINE_STEPS)
        if not getattr(self.meta, "pipeline_name", None):
            self.meta.pipeline_name = "landlord_fe_v1"
        if not self.meta.version:
            self.meta.version = (
                f"{self.meta.model_type}-seed{self.meta.seed}-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            )
        if not self.meta.created_at:
            self.meta.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved LandlordScorer -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "LandlordScorer":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LandlordScorer, got {type(obj)}")
        # Backfill pipeline fields on older artifacts
        if not getattr(obj.meta, "pipeline_steps", None):
            obj.meta.pipeline_steps = list(PIPELINE_STEPS)
        elif "preprocess" not in obj.meta.pipeline_steps:
            # Insert preprocess before predict on older exports
            steps = list(obj.meta.pipeline_steps)
            if "predict" in steps:
                i = steps.index("predict")
                steps.insert(i, "preprocess")
            else:
                steps.append("preprocess")
            obj.meta.pipeline_steps = steps
        if not getattr(obj.meta, "pipeline_name", None):
            obj.meta.pipeline_name = "landlord_fe_v1"
        return obj

    def package_manifest(self) -> dict[str, Any]:
        """
        Explicit inventory of what this artifact packs together.

        Always includes: model, preprocessing, feature_engineering, shap.
        """
        if self.meta.model_type == "catboost":
            preprocess_detail = (
                "schema alignment + categorical fill ('__missing__') + "
                "numeric coerce; CatBoost native cat handling"
            )
        else:
            preprocess_detail = (
                "schema alignment + median numeric imputer + ordinal cat encoder "
                "(fitted objects stored on this scorer)"
            )
        return {
            "components": list(PACKAGE_COMPONENTS),
            "model": {
                "included": self.model is not None,
                "type": self.meta.model_type,
                "version": self.meta.version,
            },
            "preprocessing": {
                "included": True,
                "detail": preprocess_detail,
                "has_fitted_encoders": (
                    self.num_imputer is not None or self.cat_encoder is not None
                ),
                "numeric_cols": list(self.meta.numeric_cols),
                "categorical_cols": list(self.meta.categorical_cols),
            },
            "feature_engineering": {
                "included": True,
                "pipeline_name": self.meta.pipeline_name,
                "feature_set": self.meta.feature_set,
                "steps": [
                    s for s in self.meta.pipeline_steps
                    if s not in {"preprocess", "predict", "shap_explain"}
                ],
            },
            "shap": {
                "included": True,
                "method": "TreeSHAP",
                "global_importance_n": len(self.meta.global_importance or []),
            },
            "pipeline_steps": list(self.meta.pipeline_steps),
        }

    def metadata_dict(self) -> dict[str, Any]:
        d = self.meta.to_dict()
        d["n_reference_scores"] = int(len(self.reference_scores))
        d["has_encoders"] = self.num_imputer is not None or self.cat_encoder is not None
        d["feature_glossary"] = glossary_as_records()
        # Four pillars — always true for a valid LandlordScorer export
        d["includes_model"] = self.model is not None
        d["includes_preprocessing"] = True
        d["includes_feature_engineering"] = True
        d["includes_feature_pipeline"] = True  # alias
        d["includes_shap"] = True
        d["package_components"] = list(PACKAGE_COMPONENTS)
        d["package_manifest"] = self.package_manifest()
        return d

    # ------------------------------------------------------------------
    # Preprocessing (packed)
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """
        Apply packed preprocessing to a feature matrix.

        CatBoost: column order/schema + cat fill + numeric coerce.
        Other models: same, then fitted median imputer + ordinal encoder.
        """
        return self.prepare_features(df)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Return model-ready features (DataFrame for CatBoost, ndarray otherwise)."""
        X = prepare_model_frame(
            df,
            self.meta.numeric_cols,
            self.meta.categorical_cols,
            model_type=self.meta.model_type,
        )
        if self.meta.model_type == "catboost":
            return X
        if self.num_imputer is None or self.cat_encoder is None:
            raise RuntimeError(
                f"Scorer for {self.meta.model_type} is missing fitted encoders. "
                "Re-export with build_scorer_from_artifacts()."
            )
        num = self.num_imputer.transform(X)
        cat = self.cat_encoder.transform(X)
        if num.size or cat.size:
            return np.hstack([num, cat])
        return np.empty((len(X), 0), dtype=float)

    def _prepare_shap_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature frame used for TreeExplainer (always named columns)."""
        return prepare_model_frame(
            df,
            self.meta.numeric_cols,
            self.meta.categorical_cols,
            model_type=self.meta.model_type,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        X = self.prepare_features(df)
        if self.meta.model_type == "catboost":
            from catboost import Pool

            pool = Pool(
                X,
                cat_features=self.meta.categorical_cols if self.meta.categorical_cols else None,
            )
            return np.asarray(self.model.predict(pool), dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)

    def _percentile_ranks(self, scores: np.ndarray) -> np.ndarray:
        if len(self.reference_scores) == 0:
            return np.full(len(scores), np.nan)
        ref = np.sort(self.reference_scores)
        # empirical CDF rank in [0, 100]
        ranks = np.searchsorted(ref, scores, side="right") / len(ref) * 100.0
        return ranks

    def _quality_band(self, percentile: float) -> str:
        if np.isnan(percentile):
            return "unknown"
        if percentile >= self.meta.top_percentile:
            return "good"
        if percentile <= self.meta.bottom_percentile:
            return "bad"
        return "neutral"

    def _resolve_tenant_count(self, row: pd.Series) -> int | None:
        """Prefer TenantCount; fall back to PortfolioSize (matched tenants)."""
        for col in ("TenantCount", "PortfolioSize"):
            if col in row.index and pd.notna(row[col]):
                try:
                    return int(row[col])
                except (TypeError, ValueError):
                    continue
        return None

    def _confidence(self, tenant_count: Any) -> str:
        if tenant_count is None or (isinstance(tenant_count, float) and np.isnan(tenant_count)):
            return "n/a"
        n = int(tenant_count)
        if n >= self.meta.confidence_high_threshold:
            return "high"
        if n >= self.meta.confidence_medium_threshold:
            return "medium"
        return "low"

    def _score_0_100(self, raw: float) -> float:
        """Map raw AdjustedScore-scale prediction to 0–100 via reference percentiles."""
        if len(self.reference_scores) == 0:
            return float(np.clip(raw * 100.0, 0.0, 100.0))
        pct = float(np.searchsorted(np.sort(self.reference_scores), raw, side="right")
                    / len(self.reference_scores) * 100.0)
        return float(np.clip(pct, 0.0, 100.0))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score one or more landlords.

        Returns a DataFrame with PredictedScore, PredictedQualityScore (0–100),
        PercentileRank, QualityBand, ConfidenceLevel, ModelType, ModelVersion.
        """
        if len(df) == 0:
            return pd.DataFrame()

        raw = self.predict_scores(df)
        pct = self._percentile_ranks(raw)

        rows = []
        for i in range(len(df)):
            lid = df.iloc[i]["LandLordID"] if "LandLordID" in df.columns else None
            tenants = self._resolve_tenant_count(df.iloc[i])
            p = float(pct[i]) if len(pct) else float("nan")
            conf = self._confidence(tenants)
            rows.append({
                "LandLordID": lid,
                "PredictedScore": float(raw[i]),
                "PredictedQualityScore": self._score_0_100(float(raw[i])),
                "PercentileRank": p,
                "QualityBand": self._quality_band(p),
                "TenantCount": tenants,
                "ConfidenceLevel": conf,
                "ConfidenceNote": (
                    None
                    if tenants is not None
                    else "Add TenantCount (or PortfolioSize) to estimate confidence."
                ),
                "ModelType": self.meta.model_type,
                "ModelVersion": self.meta.version,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Explain (SHAP)
    # ------------------------------------------------------------------

    def explain(
        self,
        df: pd.DataFrame,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Local TreeSHAP drivers for each row.

        Returns a list of dicts:
          LandLordID, TopPositiveDrivers, TopNegativeDrivers
        """
        import shap

        X = self._prepare_shap_frame(df)
        if self.meta.model_type == "catboost":
            explainer = shap.TreeExplainer(self.model)
            explanation = explainer(X)
        else:
            X_model = self.prepare_features(df)
            X_np = np.asarray(X_model, dtype=float)
            X_np = np.nan_to_num(X_np, nan=0.0)
            explainer = shap.TreeExplainer(self.model)
            explanation = explainer(X_np)
            try:
                explanation.feature_names = list(self.meta.numeric_cols) + list(
                    self.meta.categorical_cols
                )
            except Exception:
                pass

        out: list[dict[str, Any]] = []
        for i in range(len(df)):
            pos, neg = local_top_drivers(explanation, i, top_k=top_k)
            # Prefer values from the input row (SHAP data can be NaN for missing inputs)
            row = df.iloc[i]
            refreshed: list[dict[str, Any]] = []
            for d in pos + neg:
                feat = d.get("feature")
                if feat and feat in row.index:
                    fv = row[feat]
                    if isinstance(fv, (np.floating, float)):
                        d["feature_value"] = None if np.isnan(fv) else float(fv)
                    elif isinstance(fv, (np.integer, int)):
                        d["feature_value"] = int(fv)
                    elif pd.isna(fv):
                        d["feature_value"] = None
                    else:
                        d["feature_value"] = fv
                else:
                    d["feature_value"] = None
                refreshed.append(enrich_driver(d))
            n_pos = len(pos)
            pos = refreshed[:n_pos]
            neg = refreshed[n_pos:]

            lid = row["LandLordID"] if "LandLordID" in row.index else None
            out.append({
                "LandLordID": lid,
                "TopPositiveDrivers": pos,
                "TopNegativeDrivers": neg,
            })
        return out

    def score_with_explanation(
        self,
        df: pd.DataFrame,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Predict + local SHAP + plain-language narrative for each landlord."""
        preds = self.predict(df)
        expls = self.explain(df, top_k=top_k)
        combined: list[dict[str, Any]] = []
        for i in range(len(preds)):
            row = preds.iloc[i].to_dict()
            row.update(expls[i])
            narrative = build_interpretation(
                landlord_id=row.get("LandLordID"),
                quality_band=row.get("QualityBand"),
                quality_score=row.get("PredictedQualityScore"),
                percentile=row.get("PercentileRank"),
                confidence=row.get("ConfidenceLevel"),
                tenant_count=row.get("TenantCount"),
                predicted_score=row.get("PredictedScore"),
                positive_drivers=row.get("TopPositiveDrivers") or [],
                negative_drivers=row.get("TopNegativeDrivers") or [],
            )
            row.update(narrative)
            combined.append(row)
        return combined

    # ------------------------------------------------------------------
    # End-to-end: FE pipeline (packed) -> predict -> SHAP
    # ------------------------------------------------------------------

    def detect_input_kind(self, df: pd.DataFrame) -> str:
        """Classify upload as raw_landlords | landlord_ids | feature_matrix."""
        from src.features.inference import detect_input_kind

        return detect_input_kind(
            df,
            numeric_cols=self.meta.numeric_cols,
            categorical_cols=self.meta.categorical_cols,
        )

    def transform(
        self,
        df: pd.DataFrame,
        *,
        companies: pd.DataFrame | None = None,
        companies_scored: pd.DataFrame | None = None,
        landlords_lookup: pd.DataFrame | None = None,
        cfg: dict | None = None,
        force_pipeline: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Run the training-aligned FE pipeline packed with this artifact.

        Same path as training (minus AdjustedScore target):
        detect → resolve landlords → build_model_base → company targets →
        landlord_features → portfolio_features.

        Returns
        -------
        (feature_matrix, pipeline_info)
        """
        from src.features.inference import prepare_upload_for_scoring

        features, info = prepare_upload_for_scoring(
            df,
            companies=companies,
            companies_scored=companies_scored,
            landlords_lookup=landlords_lookup,
            numeric_cols=self.meta.numeric_cols,
            categorical_cols=self.meta.categorical_cols,
            feature_set=self.meta.feature_set,  # type: ignore[arg-type]
            cfg=cfg,
            force_pipeline=force_pipeline,
        )
        info["pipeline_name"] = self.meta.pipeline_name
        info["pipeline_steps"] = list(self.meta.pipeline_steps)
        logger.info(
            "Pipeline %s [%s]: applied=%s -> %d landlords x %d cols",
            self.meta.pipeline_name,
            info.get("input_kind"),
            info.get("pipeline_applied"),
            len(features),
            features.shape[1],
        )
        return features, info

    def score_end_to_end(
        self,
        df: pd.DataFrame,
        *,
        top_k: int = 5,
        companies: pd.DataFrame | None = None,
        companies_scored: pd.DataFrame | None = None,
        landlords_lookup: pd.DataFrame | None = None,
        cfg: dict | None = None,
        force_pipeline: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Full serving path packed in the artifact:

          feature engineering → preprocess → predict → SHAP

        Returns
        -------
        (score_rows, pipeline_info)
        """
        features, info = self.transform(
            df,
            companies=companies,
            companies_scored=companies_scored,
            landlords_lookup=landlords_lookup,
            cfg=cfg,
            force_pipeline=force_pipeline,
        )
        info["package_components"] = list(PACKAGE_COMPONENTS)
        return self.score_with_explanation(features, top_k=top_k), info


# ---------------------------------------------------------------------------
# Builder from existing training artifacts
# ---------------------------------------------------------------------------


def build_scorer_from_artifacts(
    *,
    model: Any,
    model_type: str,
    matrix: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict | None = None,
    feature_set: str = "with_portfolio",
    metrics: dict | None = None,
    global_importance: pd.DataFrame | None = None,
    reference_scores: np.ndarray | None = None,
    version: str | None = None,
) -> LandlordScorer:
    """
    Build the combined artifact: FE + preprocessing + model + SHAP metadata.

    Non-CatBoost models get fitted ``_NumImputer`` / ``_CatEncoder`` stored on
    the scorer. CatBoost keeps native categoricals; schema prep is still packed
    via ``preprocess`` / ``prepare_features``.
    """
    cfg = cfg or {}
    ls = cfg.get("landlord_score", {})
    seed = int(cfg.get("seed", 42))

    num_imputer: _NumImputer | None = None
    cat_encoder: _CatEncoder | None = None
    if model_type != "catboost":
        X_frame = prepare_model_frame(matrix, numeric_cols, categorical_cols, model_type)
        num_imputer = _NumImputer()
        cat_encoder = _CatEncoder()
        num_imputer.fit_transform(X_frame, numeric_cols)
        cat_encoder.fit_transform(X_frame, categorical_cols)

    meta = ScorerMeta(
        model_type=model_type,
        numeric_cols=list(numeric_cols),
        categorical_cols=list(categorical_cols),
        feature_set=feature_set,
        seed=seed,
        version=version or "",
        top_percentile=float(ls.get("top_percentile", 70)),
        bottom_percentile=float(ls.get("bottom_percentile", 30)),
        confidence_high_threshold=int(ls.get("confidence_high_threshold", 10)),
        confidence_medium_threshold=int(ls.get("confidence_medium_threshold", 5)),
        metrics=metrics or {},
        global_importance=(
            global_importance.to_dict(orient="records")
            if global_importance is not None and len(global_importance)
            else []
        ),
        pipeline_steps=list(PIPELINE_STEPS),
        pipeline_name="landlord_fe_v1",
    )

    scorer = LandlordScorer(
        model,
        meta,
        reference_scores=None,
        num_imputer=num_imputer,
        cat_encoder=cat_encoder,
    )

    if reference_scores is not None:
        scorer.reference_scores = np.asarray(reference_scores, dtype=float)
    else:
        logger.info("Computing reference scores on %d landlords …", len(matrix))
        scorer.reference_scores = scorer.predict_scores(matrix)

    scorer.meta.n_reference = int(len(scorer.reference_scores))
    return scorer
