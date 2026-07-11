"""
feature_glossary.py — Human-readable labels and explanations for model features.

Used by SHAP case studies, reports, and LandlordScorer so explanations show
plain-language meaning instead of raw column names alone.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Glossary: feature → {label, description, value_hint}
# ---------------------------------------------------------------------------

FEATURE_GLOSSARY: dict[str, dict[str, str]] = {
    # Landlord profile
    "LandlordAge": {
        "label": "Landlord age (years)",
        "description": "How long the landlord has been operating (years since founding).",
        "value_hint": "years",
    },
    "Area": {
        "label": "Property / campus area",
        "description": "Physical area associated with the landlord.",
        "value_hint": "area units",
    },
    "TotalPopulationAround": {
        "label": "Nearby population",
        "description": "Population living around the landlord’s location.",
        "value_hint": "people",
    },
    "ActiveCompanies": {
        "label": "Reported active companies",
        "description": "Number of companies the landlord reports as active tenants.",
        "value_hint": "companies",
    },
    "PopulationDensity": {
        "label": "Population density",
        "description": "Nearby population divided by area (crowding / urban intensity).",
        "value_hint": "people per area",
    },
    "AreaPerCompany": {
        "label": "Area per company",
        "description": "Space available per reported company (area ÷ active companies).",
        "value_hint": "area per company",
    },
    "PopulationPerCompany": {
        "label": "Population per company",
        "description": "Nearby population per reported company.",
        "value_hint": "people per company",
    },
    "PreferredIndustry": {
        "label": "Preferred industry",
        "description": "Industry focus the landlord prefers for tenants.",
        "value_hint": "category",
    },
    "OriginCity": {
        "label": "Origin city",
        "description": "City associated with the landlord’s origin / base.",
        "value_hint": "city",
    },
    "OriginCountry": {
        "label": "Origin country",
        "description": "Country associated with the landlord’s origin / base.",
        "value_hint": "country",
    },
    # Portfolio aggregates
    "PortfolioSize": {
        "label": "Matched tenant count",
        "description": "How many tenant companies were matched for this landlord in the data.",
        "value_hint": "tenants",
    },
    "PortfolioActiveRate": {
        "label": "Share of active tenants",
        "description": "Fraction of matched tenants labelled as currently active / seeking work.",
        "value_hint": "share (0–1)",
    },
    "PortfolioMeanBudget": {
        "label": "Average tenant monthly budget",
        "description": "Mean monthly budget across the landlord’s matched tenant companies.",
        "value_hint": "currency units / month",
    },
    "PortfolioStdBudget": {
        "label": "Budget variation across tenants",
        "description": "How much tenant monthly budgets vary (standard deviation).",
        "value_hint": "currency units",
    },
    "PortfolioMeanSales": {
        "label": "Average tenant sales intensity",
        "description": "Mean sales-of-main-product signal across tenant companies.",
        "value_hint": "sales index",
    },
    "PortfolioMeanClients": {
        "label": "Average tenant client base",
        "description": "Mean number of active clients that tenant companies report.",
        "value_hint": "clients",
    },
    "PortfolioMeanRetention": {
        "label": "Average tenant client retention",
        "description": "Mean share of returning clients among tenants (loyalty / stickiness).",
        "value_hint": "share (0–1)",
    },
    "PortfolioMeanSuccessScore": {
        "label": "Average tenant success score",
        "description": "Mean composite company-success score across the tenant portfolio.",
        "value_hint": "score (0–1)",
    },
    "PortfolioStdSuccessScore": {
        "label": "Success-score variation across tenants",
        "description": "How much tenant success scores vary within the portfolio.",
        "value_hint": "score units",
    },
}


def feature_label(feature: str) -> str:
    """Short human-readable name; falls back to the raw feature key."""
    entry = FEATURE_GLOSSARY.get(feature)
    return entry["label"] if entry else feature.replace("_", " ")


def feature_description(feature: str) -> str:
    entry = FEATURE_GLOSSARY.get(feature)
    if entry:
        return entry["description"]
    return f"Model feature `{feature}`."


def format_feature_value(feature: str, value: Any) -> str:
    """Format a feature value for display in explanations."""
    if value is None:
        return "not in input"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "not in input"
    if isinstance(value, (str, np.str_)):
        s = str(value)
        return "not in input" if s in {"__missing__", "nan", "None", ""} else s

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(v) or np.isinf(v):
        return "not in input"

    # Shares / rates (0–1)
    if feature in {
        "PortfolioActiveRate",
        "PortfolioMeanRetention",
        "PortfolioMeanSuccessScore",
    }:
        if 0.0 <= v <= 1.0:
            return f"{v:.1%}"
        return f"{v:.2f}"

    # Counts / sizes
    if feature in {
        "ActiveCompanies",
        "PortfolioSize",
        "PortfolioMeanClients",
        "LandlordAge",
        "TotalPopulationAround",
    }:
        if abs(v - round(v)) < 1e-6:
            return f"{int(round(v)):,}"
        return f"{v:,.1f}"

    # Money-like
    if feature in {"PortfolioMeanBudget", "PortfolioStdBudget", "Area", "AreaPerCompany"}:
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.2f}"

    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4g}"


def enrich_driver(driver: dict[str, Any]) -> dict[str, Any]:
    """
    Add human-readable fields to a SHAP driver dict.

    SHAP values are *local contributions*: how much this feature value moved
    **this landlord's prediction** away from the model's average baseline —
    not a verdict that the feature itself is "good" or "bad".

    Input keys: feature, shap_value, feature_value
    Adds: label, description, value_display, direction, explanation
    """
    feature = str(driver.get("feature", "unknown"))
    shap_value = float(driver.get("shap_value", 0.0))
    raw_value = driver.get("feature_value")
    label = feature_label(feature)
    desc = feature_description(feature)
    value_display = format_feature_value(feature, raw_value)
    direction = "up" if shap_value > 0 else "down"
    magnitude = abs(shap_value)
    missing = value_display == "not in input"
    negligible = magnitude < 0.001

    if missing:
        explanation = (
            f"{label} was not in the input. The model still applied its pattern "
            f"for missing values, which moved this landlord's predicted score "
            f"{direction} by {magnitude:.4f} versus the typical baseline."
        )
    else:
        explanation = (
            f"Because {label} was {value_display}, the model moved this "
            f"landlord's predicted score {direction} by {magnitude:.4f} "
            f"versus a typical landlord baseline."
        )
    if negligible:
        explanation += " (Very small contribution — practically negligible.)"
    explanation += f" {desc}"

    out = dict(driver)
    out.update({
        "label": label,
        "description": desc,
        "value_display": value_display,
        "direction": direction,
        "explanation": explanation,
        "value_missing": missing,
        "negligible": negligible,
        "effect_phrase": (
            f"moved prediction {direction} by {magnitude:.4f}"
            + (" (negligible)" if negligible else "")
        ),
    })
    return out


def enrich_drivers(drivers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_driver(d) for d in drivers]


def driver_phrase(driver: dict[str, Any], *, include_value: bool = True) -> str:
    """Compact phrase for narratives."""
    d = enrich_driver(driver) if "label" not in driver else driver
    # Re-enrich if value_display still says unknown from older payloads
    if "value_display" not in d or d.get("value_display") == "unknown":
        d = enrich_driver(d)
    sign = "+" if d["shap_value"] > 0 else ""
    if include_value:
        if d.get("value_missing") or d.get("value_display") == "not in input":
            return (
                f"{d['label']} (not in input, "
                f"{sign}{d['shap_value']:.4f})"
            )
        return (
            f"{d['label']} was {d['value_display']} "
            f"({sign}{d['shap_value']:.4f})"
        )
    return f"{d['label']} ({sign}{d['shap_value']:.4f})"


def glossary_as_records() -> list[dict[str, str]]:
    """Full glossary for reports / meta JSON."""
    rows = []
    for key, meta in FEATURE_GLOSSARY.items():
        rows.append({
            "feature": key,
            "label": meta["label"],
            "description": meta["description"],
            "value_hint": meta.get("value_hint", ""),
        })
    return rows
