"""
Tool 2: Client Engagement

Computes client retention, growth velocity, and visitor conversion metrics.
All calculations are deterministic Python — the LLM must not compute
these rates inline.

Metrics
-------
retention_rate          : returning_clients / todays_clients
active_client_ratio     : total_active_clients / todays_clients
client_growth_rate_6m   : annualised growth approximation from 6-month window
client_growth_rate_12m  : YoY growth rate
conversion_rate_7d      : clients_7d / visitors_7d  (recent acquisition efficiency)
visitor_trend_ratio     : visitors_7d annualised vs visitors_12m  (traffic momentum)
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    label: str,
    issues: list[str],
) -> Optional[float]:
    """Divide numerator by denominator with full null/zero safety."""
    if numerator is None:
        issues.append(f"missing: {label} numerator")
        return None
    if denominator is None:
        issues.append(f"missing: {label} denominator")
        return None
    if abs(denominator) < 1e-9:
        issues.append(f"zero: {label} denominator")
        return None
    return round(numerator / denominator, 4)


def _growth_rate(
    recent: Optional[float],
    baseline: Optional[float],
    label: str,
    issues: list[str],
) -> Optional[float]:
    """
    (recent - baseline) / baseline — standard period-over-period growth rate.
    Returns None if either value is missing or baseline is zero.
    """
    if recent is None:
        issues.append(f"missing: {label} (recent)")
        return None
    if baseline is None:
        issues.append(f"missing: {label} (baseline)")
        return None
    if abs(baseline) < 1e-9:
        issues.append(f"zero: {label} baseline")
        return None
    return round((recent - baseline) / baseline, 4)


@tool
def assess_client_engagement(
    todays_clients: Optional[float],
    returning_clients: Optional[float],
    total_active_clients: Optional[float],
    clients_7d: Optional[float],
    clients_6m: Optional[float],
    clients_12m: Optional[float],
    visitors_7d: Optional[float],
    visitors_6m: Optional[float],
    visitors_12m: Optional[float],
) -> dict:
    """
    Compute client retention, growth, and conversion metrics for a company.

    Parameters
    ----------
    todays_clients       : Number of clients active today.
    returning_clients    : Subset of today's clients who are returning (not new).
    total_active_clients : Total clients active in the broader window.
    clients_7d           : Clients acquired/active in the last 7 days.
    clients_6m           : Clients acquired/active in the last 6 months.
    clients_12m          : Clients acquired/active in the last 12 months.
    visitors_7d          : Unique visitors in the last 7 days.
    visitors_6m          : Unique visitors in the last 6 months.
    visitors_12m         : Unique visitors in the last 12 months.

    Returns
    -------
    dict with keys:
        retention_rate        : returning / today's clients (loyalty signal)
        active_client_ratio   : total_active / today's clients (engagement breadth)
        client_growth_rate_6m : (clients_6m - clients_12m/2) / (clients_12m/2)
                                Annualised approximation: compares H2 vs H1 of the year
        client_growth_rate_12m: (clients_12m - clients_6m) / clients_6m
                                YoY proxy: 12-month total vs 6-month run-rate
        conversion_rate_7d    : clients_7d / visitors_7d (recent acquisition efficiency)
        visitor_trend_ratio   : (visitors_7d * 52) / visitors_12m
                                Annualised weekly run-rate vs actual 12-month total;
                                >1 = accelerating traffic, <1 = decelerating
        data_quality_issues   : Fields that were missing or zero
    """
    issues: list[str] = []

    # --- Retention ---
    retention_rate = _safe_ratio(
        returning_clients, todays_clients, "retention_rate", issues
    )

    # --- Active client breadth ---
    active_client_ratio = _safe_ratio(
        total_active_clients, todays_clients, "active_client_ratio", issues
    )

    # --- Client growth: 6-month annualised approximation ---
    # Treats clients_12m as a full-year baseline split into two halves.
    # clients_6m is the more recent half; clients_12m/2 is the prior-half baseline.
    h1_baseline = (clients_12m / 2.0) if clients_12m is not None else None
    client_growth_rate_6m = _growth_rate(
        clients_6m, h1_baseline, "client_growth_rate_6m", issues
    )

    # --- Client growth: 12-month YoY proxy ---
    # clients_12m total vs clients_6m run-rate (doubling 6m gives expected annual).
    expected_annual = (clients_6m * 2.0) if clients_6m is not None else None
    client_growth_rate_12m = _growth_rate(
        clients_12m, expected_annual, "client_growth_rate_12m", issues
    )

    # --- Conversion rate (7-day window) ---
    conversion_rate_7d = _safe_ratio(
        clients_7d, visitors_7d, "conversion_rate_7d", issues
    )

    # --- Visitor trend: weekly run-rate vs 12-month actual ---
    annualised_weekly = (visitors_7d * 52.0) if visitors_7d is not None else None
    visitor_trend_ratio = _safe_ratio(
        annualised_weekly, visitors_12m, "visitor_trend_ratio", issues
    )

    if issues:
        logger.debug("assess_client_engagement: data quality issues — %s", issues)

    return {
        "retention_rate": retention_rate,
        "active_client_ratio": active_client_ratio,
        "client_growth_rate_6m": client_growth_rate_6m,
        "client_growth_rate_12m": client_growth_rate_12m,
        "conversion_rate_7d": conversion_rate_7d,
        "visitor_trend_ratio": visitor_trend_ratio,
        "data_quality_issues": issues,
    }
