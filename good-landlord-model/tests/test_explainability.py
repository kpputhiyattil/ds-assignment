"""
Tests for src/explainability/shap_utils.py

Pure / lightweight unit tests — no full CatBoost SHAP run required except
one optional TreeExplainer smoke test on a tiny RandomForest.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_utils import (
    build_case_studies,
    local_top_drivers,
    mean_abs_shap_table,
    prepare_model_frame,
    select_case_study_ids,
    write_explainability_report,
)


def _fake_explanation(
    n_rows: int = 10,
    feature_names: list[str] | None = None,
    seed: int = 0,
) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    feature_names = feature_names or ["f_a", "f_b", "f_c"]
    n_feat = len(feature_names)
    values = rng.normal(0, 0.05, size=(n_rows, n_feat))
    data = rng.uniform(0, 1, size=(n_rows, n_feat))
    return SimpleNamespace(
        values=values,
        data=data,
        feature_names=feature_names,
        landlord_index=list(range(n_rows)),
    )


class TestPrepareModelFrame:
    def test_orders_numeric_then_categorical(self):
        matrix = pd.DataFrame({
            "num1": [1.0, 2.0],
            "cat1": ["a", None],
            "other": [9, 9],
        })
        X = prepare_model_frame(matrix, ["num1"], ["cat1"])
        assert list(X.columns) == ["num1", "cat1"]
        assert X["cat1"].tolist() == ["a", "__missing__"]

    def test_missing_feature_raises(self):
        matrix = pd.DataFrame({"num1": [1.0]})
        with pytest.raises(ValueError, match="missing"):
            prepare_model_frame(matrix, ["num1", "gone"], [])


class TestSelectCaseStudyIds:
    def test_returns_good_bad_mid(self):
        scores = pd.DataFrame({
            "LandLordID": [f"L{i}" for i in range(20)],
            "AdjustedScore": np.linspace(0.1, 0.9, 20),
            "TenantCount": [10] * 20,
        })
        cases = select_case_study_ids(scores, n_good=2, n_bad=2, n_mid=1, min_tenants=5)
        assert set(cases["band"]) >= {"good", "bad", "neutral"}
        assert len(cases) == 5
        goods = cases[cases["band"] == "good"]["AdjustedScore"]
        bads = cases[cases["band"] == "bad"]["AdjustedScore"]
        assert goods.min() > bads.max()

    def test_falls_back_when_few_tenants(self):
        scores = pd.DataFrame({
            "LandLordID": ["A", "B", "C"],
            "AdjustedScore": [0.9, 0.5, 0.1],
            "TenantCount": [1, 1, 1],
        })
        cases = select_case_study_ids(scores, n_good=1, n_bad=1, n_mid=1, min_tenants=99)
        assert len(cases) >= 1


class TestLocalDriversAndImportance:
    def test_local_top_drivers_signs(self):
        expl = _fake_explanation(n_rows=3, feature_names=["x", "y", "z"], seed=1)
        expl.values[0] = np.array([0.2, -0.1, 0.05])
        pos, neg = local_top_drivers(expl, 0, top_k=2)
        assert pos[0]["feature"] == "x"
        assert neg[0]["feature"] == "y"
        assert all(d["shap_value"] > 0 for d in pos)
        assert all(d["shap_value"] < 0 for d in neg)

    def test_mean_abs_shap_table_sorted(self):
        expl = _fake_explanation(n_rows=20, feature_names=["a", "b", "c"], seed=2)
        table = mean_abs_shap_table(expl, top_n=3)
        assert "feature" in table.columns
        assert "label" in table.columns
        assert "mean_abs_shap" in table.columns
        assert table["mean_abs_shap"].is_monotonic_decreasing

    def test_local_drivers_include_human_fields(self):
        expl = _fake_explanation(
            n_rows=1,
            feature_names=["PortfolioMeanClients", "Area", "LandlordAge"],
            seed=4,
        )
        expl.values[0] = np.array([0.02, -0.01, 0.005])
        expl.data[0] = np.array([12.5, 100.0, 8.0])
        pos, neg = local_top_drivers(expl, 0, top_k=2)
        assert pos[0]["feature"] == "PortfolioMeanClients"
        assert pos[0]["label"] == "Average tenant client base"
        assert "explanation" in pos[0]
        assert "increased" in pos[0]["explanation"]
        assert neg[0]["direction"] == "decreased"


class TestBuildCaseStudiesAndReport:
    def test_build_case_studies_without_plots(self, tmp_path, monkeypatch):
        # Avoid matplotlib/shap plot calls: patch waterfall to a no-op path writer
        from src.explainability import shap_utils as su

        def _fake_waterfall(explanation, row_pos, out_path, **kwargs):
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
            return p

        monkeypatch.setattr(su, "plot_local_waterfall", _fake_waterfall)

        matrix = pd.DataFrame({
            "LandLordID": [f"L{i}" for i in range(5)],
            "AdjustedScore": [0.8, 0.7, 0.5, 0.3, 0.2],
            "TenantCount": [10] * 5,
        })
        expl = _fake_explanation(n_rows=5, seed=3)
        expl.landlord_index = list(matrix.index)

        case_rows = pd.DataFrame([
            {"LandLordID": "L0", "band": "good", "AdjustedScore": 0.8, "TenantCount": 10},
            {"LandLordID": "L4", "band": "bad", "AdjustedScore": 0.2, "TenantCount": 10},
        ])
        studies = build_case_studies(
            matrix, expl, case_rows, figures_dir=tmp_path / "cases"
        )
        assert len(studies) == 2
        assert studies[0]["band"] == "good"
        assert studies[0]["waterfall_path"] is not None
        assert "What pushed the score up" in studies[0]["narrative"]

    def test_write_explainability_report(self, tmp_path):
        importance = pd.DataFrame({
            "feature": ["PortfolioMeanClients", "Area"],
            "mean_abs_shap": [0.01, 0.001],
        })
        cases = [{
            "LandLordID": "L1",
            "band": "good",
            "AdjustedScore": 0.5,
            "oof_pred": 0.48,
            "TenantCount": 8,
            "top_positive_drivers": [
                {"feature": "PortfolioMeanClients", "shap_value": 0.02, "feature_value": 5.0}
            ],
            "top_negative_drivers": [],
            "waterfall_path": None,
            "narrative": "Test narrative.",
        }]
        out = write_explainability_report(
            model_type="catboost",
            global_importance=importance,
            case_studies=cases,
            figure_paths={},
            out_path=tmp_path / "shap_report.md",
            n_shap_rows=100,
            n_landlords=1000,
        )
        text = out.read_text(encoding="utf-8")
        assert "SHAP Explainability Report" in text
        assert "catboost" in text
        assert "PortfolioMeanClients" in text
        assert "L1" in text


@pytest.mark.slow
class TestTreeExplainerSmoke:
    """Optional smoke test — skipped if sklearn RF + shap unavailable."""

    def test_compute_shap_on_tiny_rf(self):
        pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestRegressor

        from src.explainability.shap_utils import compute_shap_explanation

        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "x1": rng.normal(size=40),
            "x2": rng.normal(size=40),
        })
        y = X["x1"] * 0.5 + rng.normal(scale=0.1, size=40)
        model = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=3)
        model.fit(X, y)

        expl = compute_shap_explanation(
            model, X, model_type="random_forest", max_samples=20, seed=0
        )
        assert len(expl.values) == 20
        assert expl.values.shape[1] == 2
