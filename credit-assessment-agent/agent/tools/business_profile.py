"""
Tool 3: Business Profile

Surfaces qualitative risk signals from categorical and temporal fields.
Pre-computes interpretive signals (risk tier, rank band, age, status) so
the LLM receives structured inputs rather than raw strings to reason about.

After loading Companies.parquet, run:
    python -c "import pandas as pd; df = pd.read_parquet('Companies.parquet'); print(df['PrimaryType'].value_counts())"
and add any unlisted types to BUSINESS_TYPE_RISK_MAP with an appropriate tier.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk tier lookup — extend with actual PrimaryType values from the dataset.
# Unknown types default to "medium" with a logged warning.
# Tiers: "low" | "medium" | "high"
# ---------------------------------------------------------------------------
BUSINESS_TYPE_RISK_MAP: dict[str, str] = {
    # --- Low risk: established, stable, essential services ---
    "Retail": "low",
    "retail": "low",
    "Grocery": "low",
    "grocery": "low",
    "Supermarket": "low",
    "supermarket": "low",
    "Food": "low",
    "food": "low",
    "Restaurant": "low",
    "restaurant": "low",
    "Cafe": "low",
    "cafe": "low",
    "Healthcare": "low",
    "healthcare": "low",
    "Pharmacy": "low",
    "pharmacy": "low",
    "Education": "low",
    "education": "low",
    "Professional Services": "low",
    "professional_services": "low",
    "Law": "low",
    "Accounting": "low",
    "accounting": "low",
    "Logistics": "low",
    "logistics": "low",
    "Transportation": "low",
    "transportation": "low",
    "Manufacturing": "low",
    "manufacturing": "low",
    "Agriculture": "low",
    "agriculture": "low",
    "Utilities": "low",
    "utilities": "low",
    # --- Medium risk: growth-oriented, some volatility ---
    "Technology": "medium",
    "technology": "medium",
    "Tech": "medium",
    "tech": "medium",
    "Software": "medium",
    "software": "medium",
    "E-commerce": "medium",
    "ecommerce": "medium",
    "e-commerce": "medium",
    "Online Retail": "medium",
    "online_retail": "medium",
    "Media": "medium",
    "media": "medium",
    "Marketing": "medium",
    "marketing": "medium",
    "Consulting": "medium",
    "consulting": "medium",
    "Real Estate": "medium",
    "real_estate": "medium",
    "Construction": "medium",
    "construction": "medium",
    "Import/Export": "medium",
    "import_export": "medium",
    "Wholesale": "medium",
    "wholesale": "medium",
    "Events": "medium",
    "events": "medium",
    "Entertainment": "medium",
    "entertainment": "medium",
    "Hospitality": "medium",
    "hospitality": "medium",
    "Travel": "medium",
    "travel": "medium",
    "Fitness": "medium",
    "fitness": "medium",
    "Beauty": "medium",
    "beauty": "medium",
    # --- Dataset-specific PrimaryType values (Companies.parquet) ---
    "Pets": "medium",
    "Toys": "medium",
    "Office": "low",
    "Food": "low",
    "Gardening": "low",
    "Sport": "medium",
    "Furniture": "medium",
    "Electronics": "medium",
    "Pharma": "medium",
    "Banking": "medium",
    "Alcohol": "high",
    "Angel (individual)": "high",
    "Angel Group": "high",
    "Family Office": "medium",
    "VC-Backed Company": "medium",
    "Growth/Expansion": "medium",
    "Limited Partner": "medium",
    "Impact Investing": "medium",
    "Holding Company": "medium",
    "Lender/Debt Provider": "medium",
    "Other": "medium",
    # --- High risk: highly volatile, regulatory exposure, or unproven models ---
    "Startup": "high",
    "startup": "high",
    "Crypto": "high",
    "crypto": "high",
    "Cryptocurrency": "high",
    "cryptocurrency": "high",
    "NFT": "high",
    "nft": "high",
    "Gambling": "high",
    "gambling": "high",
    "Gaming": "high",
    "gaming": "high",
    "Nightclub": "high",
    "nightclub": "high",
    "Bar": "high",
    "bar": "high",
    "Speculative": "high",
    "speculative": "high",
    "Mining": "high",
    "mining": "high",
    "Cannabis": "high",
    "cannabis": "high",
    "Firearms": "high",
    "firearms": "high",
}

# Status normalisation — maps raw dataset values to canonical flags.
# Includes both generic labels and the actual Companies.parquet values.
STATUS_NORMALISATION: dict[str, str] = {
    # Active variants (generic)
    "active": "active",
    "Active": "active",
    "ACTIVE": "active",
    "open": "active",
    "Open": "active",
    "operating": "active",
    "Operating": "active",
    # Active variants (dataset-specific)
    "Actively Seeking New Employees": "active",
    "Will Consider New Projects": "active",
    "Acquired/Merged (Operating Subsidiary)": "active",
    # Inactive variants (generic)
    "inactive": "inactive",
    "Inactive": "inactive",
    "INACTIVE": "inactive",
    "dormant": "inactive",
    "Dormant": "inactive",
    "paused": "inactive",
    "Paused": "inactive",
    # Inactive variants (dataset-specific)
    "Not Making New Products": "inactive",
    "Reducing Activity": "inactive",
    "Acquired/Merged": "inactive",
    # Suspended variants
    "suspended": "suspended",
    "Suspended": "suspended",
    "SUSPENDED": "suspended",
    "on hold": "suspended",
    "On Hold": "suspended",
    "frozen": "suspended",
    # Closed variants (generic)
    "closed": "closed",
    "Closed": "closed",
    "CLOSED": "closed",
    "dissolved": "closed",
    "Dissolved": "closed",
    "liquidated": "closed",
    "Liquidated": "closed",
    "bankrupt": "closed",
    "Bankrupt": "closed",
    "terminated": "closed",
    "Terminated": "closed",
    # Closed variants (dataset-specific)
    "Out of Business": "closed",
}

_CURRENT_YEAR = datetime.now().year


def _normalise_status(raw: Optional[str]) -> tuple[str, list[str]]:
    """Map raw CompanyStatus to a canonical flag. Returns (flag, issues)."""
    issues: list[str] = []
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        issues.append("missing: company_status")
        return "unknown", issues

    normalised = STATUS_NORMALISATION.get(raw)
    if normalised is None:
        logger.warning("Unknown CompanyStatus value '%s' — defaulting to 'unknown'", raw)
        issues.append(f"unknown_status_value: '{raw}'")
        return "unknown", issues

    return normalised, issues


def _risk_tier(primary_type: Optional[str]) -> tuple[str, list[str]]:
    """Look up business type risk tier. Returns (tier, issues)."""
    issues: list[str] = []
    if primary_type is None:
        issues.append("missing: primary_type")
        return "medium", issues

    tier = BUSINESS_TYPE_RISK_MAP.get(primary_type)
    if tier is None:
        logger.warning(
            "Unknown PrimaryType '%s' — defaulting to 'medium' risk tier. "
            "Add it to BUSINESS_TYPE_RISK_MAP in business_profile.py.",
            primary_type,
        )
        issues.append(f"unknown_primary_type: '{primary_type}' (defaulted to medium)")
        return "medium", issues

    return tier, issues


def _rank_band(rank: Optional[float]) -> str:
    """Convert numeric rank to a qualitative band."""
    if rank is None:
        return "unknown"
    if rank > 4.0:
        return "strong"
    if rank >= 3.0:
        return "acceptable"
    return "weak"


@tool
def check_business_profile(
    company_status: Optional[str],
    primary_type: Optional[str],
    year_founded: Optional[float],
    origin_country: Optional[str],
    rank: Optional[float],
) -> dict:
    """
    Surface qualitative risk signals from categorical and temporal company fields.

    Parameters
    ----------
    company_status : Raw operational status string (e.g. 'active', 'closed').
    primary_type   : Business category/type (e.g. 'Retail', 'Technology').
    year_founded   : Year the company was established (e.g. 2015).
    origin_country : Country of incorporation or primary operation.
    rank           : Numeric rating/score for the company (scale varies by dataset).

    Returns
    -------
    dict with keys:
        status_flag            : Normalised status — 'active'|'inactive'|'suspended'|'closed'|'unknown'
        business_type_risk_tier: 'low'|'medium'|'high' based on PrimaryType lookup
        company_age_years      : Current year minus year_founded
        rank_band              : 'strong' (>4.0) | 'acceptable' (3.0–4.0) | 'weak' (<3.0) | 'unknown'
        country                : Passed through as-is for LLM context
        data_quality_issues    : Fields that were missing, zero, or unrecognised
    """
    issues: list[str] = []

    status_flag, status_issues = _normalise_status(company_status)
    issues.extend(status_issues)

    business_type_risk_tier, type_issues = _risk_tier(primary_type)
    issues.extend(type_issues)

    if year_founded is None:
        issues.append("missing: year_founded")
        company_age_years = None
    else:
        company_age_years = _CURRENT_YEAR - int(year_founded)
        if company_age_years < 0:
            issues.append(f"invalid: year_founded ({year_founded}) is in the future")
            company_age_years = None

    if origin_country is None:
        issues.append("missing: origin_country")

    if rank is None:
        issues.append("missing: rank")

    rank_band = _rank_band(rank)

    if issues:
        logger.debug("check_business_profile: data quality issues — %s", issues)

    return {
        "status_flag": status_flag,
        "business_type_risk_tier": business_type_risk_tier,
        "company_age_years": company_age_years,
        "rank_band": rank_band,
        "country": origin_country,
        "data_quality_issues": issues,
    }
