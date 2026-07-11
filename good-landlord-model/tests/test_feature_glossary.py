"""
Tests for human-readable feature glossary used in SHAP explanations.
"""
from __future__ import annotations

from src.explainability.feature_glossary import (
    FEATURE_GLOSSARY,
    driver_phrase,
    enrich_driver,
    feature_label,
    format_feature_value,
    glossary_as_records,
)


class TestGlossary:
    def test_known_feature_label(self):
        assert "client" in feature_label("PortfolioMeanClients").lower()

    def test_unknown_feature_fallback(self):
        assert feature_label("SomeNewFeat") == "SomeNewFeat"

    def test_format_missing(self):
        assert format_feature_value("PortfolioMeanClients", None) == "not in input"
        assert format_feature_value("PortfolioMeanClients", float("nan")) == "not in input"

    def test_format_share(self):
        assert format_feature_value("PortfolioActiveRate", 0.75) == "75.0%"

    def test_format_count(self):
        assert format_feature_value("PortfolioSize", 29) == "29"

    def test_enrich_driver_explanation(self):
        d = enrich_driver({
            "feature": "PortfolioMeanBudget",
            "shap_value": 0.008,
            "feature_value": 6229.12,
        })
        assert d["label"] == "Average tenant monthly budget"
        assert d["direction"] == "up"
        assert "6229" in d["value_display"] or "6,229" in d["value_display"]
        assert "moved this landlord's predicted score up" in d["explanation"]
        assert "baseline" in d["explanation"]

    def test_driver_phrase(self):
        phrase = driver_phrase({
            "feature": "PortfolioMeanClients",
            "shap_value": 0.0275,
            "feature_value": 10.86,
        })
        assert "Average tenant client base" in phrase
        assert "+0.0275" in phrase

    def test_glossary_covers_core_features(self):
        keys = set(FEATURE_GLOSSARY)
        for feat in (
            "PortfolioMeanClients",
            "PortfolioMeanBudget",
            "PortfolioActiveRate",
            "LandlordAge",
            "PreferredIndustry",
        ):
            assert feat in keys

    def test_glossary_as_records(self):
        rows = glossary_as_records()
        assert len(rows) == len(FEATURE_GLOSSARY)
        assert {"feature", "label", "description"} <= set(rows[0])
