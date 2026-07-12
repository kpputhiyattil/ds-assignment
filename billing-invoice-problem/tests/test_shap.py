"""Tests for SHAP explainability helpers (Step 7)."""

from __future__ import annotations

import numpy as np
import shap
from lightgbm import LGBMClassifier

from src.config import get_config
from src.explainability.shap_utils import (
    _confidence,
    _fmt_value,
    _tier,
    explain_row,
    format_briefing,
)
from src.features.build import feature_columns

CFG = get_config()
FEATS = feature_columns()


def test_fmt_value_flags_and_numbers() -> None:
    assert _fmt_value("is_monthly", 1.0) == "yes"
    assert _fmt_value("is_monthly", 0.0) == "no"
    assert _fmt_value("recency_days", 42.0) == "42"
    assert _fmt_value("mean_amount", 1234.5) == "1,234.50"


def test_tier_and_confidence() -> None:
    assert _tier(0.9, CFG) == "High risk"
    assert _tier(0.1, CFG) == "Low risk"
    assert _tier(0.5, CFG) == "Medium risk"
    assert _confidence(0.99).startswith("High")
    assert _confidence(0.5).startswith("Low")


def _tiny_model():
    rng = np.random.default_rng(0)
    n, d = 300, len(FEATS)
    X = rng.normal(size=(n, d))
    # Make one feature drive the label so SHAP has real signal.
    y = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)
    model = LGBMClassifier(n_estimators=40, num_leaves=15, random_state=0, verbose=-1)
    model.fit(X, y)
    return model, X


def test_explain_row_structure_and_sorting() -> None:
    model, X = _tiny_model()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    prob = float(model.predict_proba(X[:1])[0, 1])
    b = explain_row(explainer, X[0], FEATS, prob, CFG, top_k=4)

    assert set(b) >= {"churn_probability", "tier", "confidence",
                      "main_reason", "risk_increasing", "risk_decreasing"}
    # All increasing shap > 0, decreasing < 0.
    assert all(it["shap"] > 0 for it in b["risk_increasing"])
    assert all(it["shap"] < 0 for it in b["risk_decreasing"])
    # Buckets sorted by |shap| descending.
    inc = [abs(it["shap"]) for it in b["risk_increasing"]]
    assert inc == sorted(inc, reverse=True)


def test_format_briefing_renders() -> None:
    model, X = _tiny_model()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    prob = float(model.predict_proba(X[:1])[0, 1])
    b = explain_row(explainer, X[0], FEATS, prob, CFG)
    text = format_briefing("CUST_1", b)
    assert "P(churn)" in text
    assert "Main reason" in text
    assert "increasing churn risk" in text
