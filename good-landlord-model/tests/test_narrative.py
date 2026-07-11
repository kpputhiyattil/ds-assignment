"""Tests for plain-language SHAP narrative."""
from __future__ import annotations

from src.explainability.narrative import build_interpretation, signal_bullet


def test_signal_bullet_missing_strong():
    b = signal_bullet(
        {
            "feature": "PortfolioActiveRate",
            "shap_value": -0.0837,
            "feature_value": None,
            "value_display": "not in input",
            "value_missing": True,
            "label": "Share of active tenants",
        },
        positive=False,
    )
    assert b is not None
    assert "Missing" in b
    assert "active tenants" in b.lower()
    assert "reduced" in b


def test_signal_bullet_slight_positive():
    b = signal_bullet(
        {
            "feature": "PortfolioMeanClients",
            "shap_value": 0.0115,
            "feature_value": 1.0,
            "value_display": "1",
            "label": "Average tenant client base",
        },
        positive=True,
    )
    assert "Tenant client base" in b
    assert "improved" in b


def test_build_interpretation_landlord_47_style():
    out = build_interpretation(
        landlord_id="LANDLORD_0047",
        quality_band="bad",
        quality_score=28.5,
        percentile=28.5,
        confidence="low",
        tenant_count=4,
        positive_drivers=[
            {"feature": "PortfolioMeanClients", "shap_value": 0.0115, "feature_value": 1},
            {"feature": "PortfolioMeanBudget", "shap_value": 0.0054, "feature_value": None},
            {"feature": "PortfolioMeanSales", "shap_value": 0.0050, "feature_value": 0},
        ],
        negative_drivers=[
            {"feature": "PortfolioActiveRate", "shap_value": -0.0837, "feature_value": None},
            {"feature": "PortfolioMeanRetention", "shap_value": -0.0053, "feature_value": 1.0},
            {"feature": "PortfolioStdSuccessScore", "shap_value": -0.0038, "feature_value": None},
        ],
    )
    assert "missing" in out["MainReason"].lower()
    assert "active tenants" in out["MainReason"].lower()
    assert any("client base" in s.lower() for s in out["PositiveSignals"])
    assert any("Missing" in s and "active" in s.lower() for s in out["NegativeSignals"])
    assert any("tenant count" in s.lower() for s in out["NegativeSignals"])
    assert "confidence is low" in out["FinalInterpretation"].lower()
    assert "LANDLORD_0047" in out["NarrativeText"]
    assert "Main reason:" in out["NarrativeText"]
