"""
Tests for src/models/scorer.py — LandlordScorer artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.scorer import LandlordScorer, ScorerMeta, build_scorer_from_artifacts


def _tiny_matrix(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LandLordID": [f"L{i}" for i in range(n)],
        "TenantCount": rng.integers(1, 20, n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c"], size=n),
        "AdjustedScore": rng.uniform(0.2, 0.6, n),
    })


@pytest.fixture
def rf_scorer(tmp_path) -> LandlordScorer:
    matrix = _tiny_matrix()
    numeric_cols = ["x1", "x2"]
    categorical_cols = ["cat"]
    # Fit a tiny RF on encoded features via build helper
    from src.explainability.shap_utils import prepare_model_frame
    from src.models.train import _CatEncoder, _NumImputer

    Xf = prepare_model_frame(matrix, numeric_cols, categorical_cols, "random_forest")
    num = _NumImputer().fit_transform(Xf, numeric_cols)
    cat = _CatEncoder().fit_transform(Xf, categorical_cols)
    X = np.hstack([num, cat])
    y = matrix["AdjustedScore"].to_numpy()
    model = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=0)
    model.fit(X, y)

    return build_scorer_from_artifacts(
        model=model,
        model_type="random_forest",
        matrix=matrix,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cfg={
            "seed": 0,
            "landlord_score": {
                "top_percentile": 70,
                "bottom_percentile": 30,
                "confidence_high_threshold": 10,
                "confidence_medium_threshold": 5,
            },
        },
        feature_set="landlord_only",
        metrics={"mae": 0.01},
        reference_scores=y,
        version="test-rf-v1",
    )


class TestLandlordScorerPredict:
    def test_predict_columns(self, rf_scorer):
        df = _tiny_matrix(5)
        out = rf_scorer.predict(df)
        for col in (
            "LandLordID",
            "PredictedScore",
            "PredictedQualityScore",
            "PercentileRank",
            "QualityBand",
            "ConfidenceLevel",
            "ModelType",
            "ModelVersion",
        ):
            assert col in out.columns
        assert len(out) == 5
        assert set(out["QualityBand"]).issubset({"good", "neutral", "bad", "unknown"})

    def test_confidence_from_tenants(self, rf_scorer):
        df = _tiny_matrix(3)
        df.loc[0, "TenantCount"] = 15
        df.loc[1, "TenantCount"] = 7
        df.loc[2, "TenantCount"] = 2
        out = rf_scorer.predict(df)
        assert out.iloc[0]["ConfidenceLevel"] == "high"
        assert out.iloc[1]["ConfidenceLevel"] == "medium"
        assert out.iloc[2]["ConfidenceLevel"] == "low"

    def test_confidence_na_without_tenant_cols(self, rf_scorer):
        df = _tiny_matrix(1).drop(columns=["TenantCount"])
        out = rf_scorer.predict(df)
        # PortfolioSize not in tiny matrix → n/a
        assert out.iloc[0]["ConfidenceLevel"] == "n/a"
        assert out.iloc[0]["ConfidenceNote"]


class TestLandlordScorerExplain:
    def test_explain_drivers(self, rf_scorer):
        pytest.importorskip("shap")
        df = _tiny_matrix(2)
        expl = rf_scorer.explain(df, top_k=2)
        assert len(expl) == 2
        assert "TopPositiveDrivers" in expl[0]
        assert "TopNegativeDrivers" in expl[0]

    def test_score_with_explanation(self, rf_scorer):
        pytest.importorskip("shap")
        df = _tiny_matrix(2)
        rows = rf_scorer.score_with_explanation(df, top_k=2)
        assert len(rows) == 2
        assert "PredictedScore" in rows[0]
        assert "TopPositiveDrivers" in rows[0]
        assert "ExplanationSummary" in rows[0]
        assert rows[0]["ModelVersion"] == "test-rf-v1"
        if rows[0]["TopPositiveDrivers"]:
            assert "label" in rows[0]["TopPositiveDrivers"][0]
            assert "explanation" in rows[0]["TopPositiveDrivers"][0]


class TestPersistence:
    def test_save_load_roundtrip(self, rf_scorer, tmp_path):
        path = tmp_path / "scorer.joblib"
        rf_scorer.save(path)
        loaded = LandlordScorer.load(path)
        df = _tiny_matrix(4)
        a = rf_scorer.predict_scores(df)
        b = loaded.predict_scores(df)
        np.testing.assert_allclose(a, b, rtol=1e-6)
        assert loaded.meta.version == rf_scorer.meta.version

    def test_load_wrong_type_raises(self, tmp_path):
        path = tmp_path / "not_scorer.joblib"
        import joblib

        joblib.dump({"x": 1}, path)
        with pytest.raises(TypeError):
            LandlordScorer.load(path)


class TestMeta:
    def test_metadata_dict(self, rf_scorer):
        d = rf_scorer.metadata_dict()
        assert d["model_type"] == "random_forest"
        assert d["has_encoders"] is True
        assert d["n_reference_scores"] > 0
        assert d["includes_model"] is True
        assert d["includes_preprocessing"] is True
        assert d["includes_feature_engineering"] is True
        assert d["includes_feature_pipeline"] is True
        assert d["includes_shap"] is True
        assert d["package_components"] == [
            "model",
            "preprocessing",
            "feature_engineering",
            "shap",
        ]
        assert d["pipeline_name"] == "landlord_fe_v1"
        assert "detect_input_kind" in d["pipeline_steps"]
        assert "preprocess" in d["pipeline_steps"]
        assert "shap_explain" in d["pipeline_steps"]

    def test_package_manifest(self, rf_scorer):
        m = rf_scorer.package_manifest()
        for key in ("model", "preprocessing", "feature_engineering", "shap"):
            assert m[key]["included"] is True
        assert m["preprocessing"]["has_fitted_encoders"] is True
        assert m["shap"]["method"] == "TreeSHAP"


class TestPipelinePacked:
    def test_transform_feature_matrix_passthrough(self, rf_scorer):
        df = _tiny_matrix(3)
        # Tiny matrix looks like feature_matrix only if enough model cols present;
        # with schema matching numeric/categorical, detect may vary — force off
        features, info = rf_scorer.transform(df)
        assert len(features) == 3
        assert "pipeline_name" in info
        assert info["pipeline_name"] == "landlord_fe_v1"

    def test_score_end_to_end_on_feature_like_input(self, rf_scorer):
        pytest.importorskip("shap")
        df = _tiny_matrix(2)
        rows, info = rf_scorer.score_end_to_end(df, top_k=2)
        assert len(rows) == 2
        assert "PredictedScore" in rows[0]
        assert "pipeline_steps" in info
