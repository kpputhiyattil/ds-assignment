"""
narrative.py — Plain-language interpretation of SHAP drivers + score metadata.

Turns technical SHAP contributions into a short briefing:

  Main reason / Positive signals / Negative signals / Final interpretation
"""
from __future__ import annotations

from typing import Any

from src.explainability.feature_glossary import enrich_driver, feature_label


# Short names for bullets (less jargon than full glossary labels)
_SHORT: dict[str, str] = {
    "PortfolioMeanClients": "Tenant client base",
    "PortfolioMeanBudget": "Tenant monthly budget",
    "PortfolioMeanSales": "Tenant sales intensity",
    "PortfolioMeanRetention": "Tenant client retention",
    "PortfolioActiveRate": "Share of active tenants",
    "PortfolioMeanSuccessScore": "Average tenant success score",
    "PortfolioStdSuccessScore": "Success-score variation",
    "PortfolioStdBudget": "Budget variation across tenants",
    "PortfolioSize": "Matched tenant count",
    "ActiveCompanies": "Reported active companies",
    "LandlordAge": "Landlord age",
    "Area": "Property area",
    "TotalPopulationAround": "Nearby population",
    "PopulationDensity": "Population density",
    "AreaPerCompany": "Area per company",
    "PopulationPerCompany": "Population per company",
    "PreferredIndustry": "Preferred industry",
    "OriginCity": "Origin city",
    "OriginCountry": "Origin country",
}


def _short_label(feature: str, fallback_label: str | None = None) -> str:
    return _SHORT.get(feature, fallback_label or feature_label(feature))


def _strength(magnitude: float) -> str:
    if magnitude >= 0.05:
        return "strongly"
    if magnitude >= 0.02:
        return "clearly"
    return "slightly"


def _is_missing(driver: dict[str, Any]) -> bool:
    if driver.get("value_missing"):
        return True
    vd = driver.get("value_display")
    return vd in {None, "not in input", "unknown"}


def _ensure(driver: dict[str, Any]) -> dict[str, Any]:
    if "label" in driver and "value_display" in driver and "shap_value" in driver:
        return driver
    return enrich_driver(driver)


def signal_bullet(driver: dict[str, Any], *, positive: bool) -> str | None:
    """One plain-language bullet for a SHAP driver."""
    d = _ensure(driver)
    mag = abs(float(d.get("shap_value", 0.0)))
    if mag < 0.001:
        return None  # skip noise
    short = _short_label(str(d.get("feature", "")), d.get("label"))
    strength = _strength(mag)
    missing = _is_missing(d)

    if missing:
        if positive:
            return f"{short} {strength} improved the score."
        verb = "strongly reduced" if mag >= 0.05 else f"{strength} reduced"
        return f"Missing {short.lower()} {verb} the score."

    if positive:
        return f"{short} {strength} improved the score."
    return f"{short} {strength} reduced the score."


def _main_reason(
    *,
    band: str,
    dominant: dict[str, Any] | None,
    neg: list[dict[str, Any]],
    pos: list[dict[str, Any]],
) -> str:
    band_l = (band or "unknown").lower()
    classified = (
        f"The model classified this landlord as {band_l}"
        if band_l in {"good", "bad", "neutral"}
        else "The model scored this landlord"
    )

    if dominant is None:
        return f"{classified}, but no strong feature drivers stood out."

    d = _ensure(dominant)
    mag = abs(float(d["shap_value"]))
    short = _short_label(str(d.get("feature", "")), d.get("label"))
    missing = _is_missing(d)
    toward_down = float(d["shap_value"]) < 0

    # Missing tenant-activity dominates bad / weak scores
    missing_negs = [x for x in neg if _is_missing(_ensure(x))]
    if missing_negs and (band_l == "bad" or (toward_down and mag >= 0.03)):
        top_miss = _ensure(max(missing_negs, key=lambda x: abs(float(x["shap_value"]))))
        miss_short = _short_label(str(top_miss.get("feature", "")), top_miss.get("label"))
        return (
            f"{classified} mainly because important tenant-activity information "
            f"was missing, especially {miss_short.lower()}."
        )

    if missing and toward_down:
        return (
            f"{classified} mainly because {short.lower()} was missing "
            f"from the available tenant signals."
        )

    if missing and not toward_down:
        return (
            f"{classified}; a missing-value pattern for {short.lower()} "
            f"still nudged the score upward, but other factors dominate."
        )

    value = d.get("value_display", "—")
    if toward_down:
        return (
            f"{classified} mainly because {short.lower()} "
            f"({value}) weighed the score down."
        )
    return (
        f"{classified} mainly because {short.lower()} "
        f"({value}) supported a higher score."
    )


def _final_interpretation(
    *,
    band: str,
    confidence: str | None,
    tenant_count: int | None,
    neg: list[dict[str, Any]],
    pos: list[dict[str, Any]],
) -> str:
    band_l = (band or "unknown").lower()
    conf = (confidence or "n/a").lower()
    n_missing = sum(1 for d in (pos + neg) if _is_missing(_ensure(d)))
    small_n = tenant_count is not None and tenant_count < 5

    appearance = {
        "bad": "appears weak based on available tenant signals",
        "good": "appears strong based on available tenant signals",
        "neutral": "looks mid-range based on available tenant signals",
    }.get(band_l, "has a mixed score based on available signals")

    parts = [f"This landlord {appearance}"]

    if conf in {"low", "n/a"} or small_n or n_missing >= 2:
        reasons = []
        if conf == "low" or small_n:
            if tenant_count is not None:
                reasons.append(f"only {tenant_count} matched tenant(s)")
            else:
                reasons.append("limited tenant coverage")
        if n_missing:
            reasons.append("incomplete tenant information")
        if reasons:
            parts.append(
                "but the confidence is low because the model has "
                + " and ".join(reasons)
            )
        else:
            parts.append("but confidence is limited")
    elif conf == "medium":
        parts.append("with moderate confidence from the tenant sample size")
    elif conf == "high":
        parts.append("with relatively high confidence from the tenant sample size")

    return ", ".join(parts) + "."


def build_interpretation(
    *,
    landlord_id: Any = None,
    quality_band: str | None = None,
    quality_score: float | None = None,
    percentile: float | None = None,
    confidence: str | None = None,
    tenant_count: int | None = None,
    predicted_score: float | None = None,
    positive_drivers: list[dict[str, Any]] | None = None,
    negative_drivers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build the human briefing the UI should lead with.
    """
    pos = [_ensure(d) for d in (positive_drivers or [])]
    neg = [_ensure(d) for d in (negative_drivers or [])]

    pos_bullets = [b for d in pos if (b := signal_bullet(d, positive=True))]
    neg_bullets = [b for d in neg if (b := signal_bullet(d, positive=False))]

    # Confidence / sample-size as an explicit negative-context signal when low
    if confidence and confidence.lower() == "low":
        if tenant_count is not None and tenant_count < 5:
            neg_bullets.append(
                f"Small tenant count ({tenant_count}) reduced confidence."
            )
        elif "Small tenant count" not in " ".join(neg_bullets):
            neg_bullets.append("Limited tenant information reduced confidence.")

    all_d = pos + neg
    dominant = (
        max(all_d, key=lambda d: abs(float(d.get("shap_value", 0.0))))
        if all_d
        else None
    )

    band = quality_band or "unknown"
    main = _main_reason(band=band, dominant=dominant, neg=neg, pos=pos)
    final = _final_interpretation(
        band=band,
        confidence=confidence,
        tenant_count=tenant_count,
        neg=neg,
        pos=pos,
    )

    # Headline block matching the user's preferred briefing style
    lines = [
        f"Landlord ID: {landlord_id}" if landlord_id is not None else None,
        f"Prediction: {band.capitalize()}" if band != "unknown" else "Prediction: Unknown",
    ]
    if quality_score is not None and not (isinstance(quality_score, float) and quality_score != quality_score):
        lines.append(f"Quality Score: {float(quality_score):.1f} / 100")
    if percentile is not None and not (isinstance(percentile, float) and percentile != percentile):
        lines.append(f"Percentile: {float(percentile):.1f}")
    if confidence:
        lines.append(f"Confidence: {str(confidence).capitalize()}")

    lines.extend(["", "Main reason:", main, ""])
    if pos_bullets:
        lines.append("Positive signals:")
        lines.extend(f"- {b}" for b in pos_bullets[:5])
        lines.append("")
    if neg_bullets:
        lines.append("Negative signals:")
        lines.extend(f"- {b}" for b in neg_bullets[:5])
        lines.append("")
    lines.extend(["Final interpretation:", final])

    narrative_text = "\n".join(x for x in lines if x is not None)

    return {
        "MainReason": main,
        "PositiveSignals": pos_bullets[:5],
        "NegativeSignals": neg_bullets[:5],
        "FinalInterpretation": final,
        "NarrativeText": narrative_text,
        "ExplanationSummary": main,
    }
