"""
Tool 1: Financial Health

Computes derived financial metrics from raw company fields.
All arithmetic is deterministic Python — the LLM must not attempt
to compute these ratios inline.

Metrics
-------
revenue_estimate        : sales_main + sales_other
budget_utilisation_ratio: revenue / monthly_budget  (>1 = revenue covers budget)
revenue_per_investment  : revenue / investments
sales_mix_ratio         : sales_main / (sales_main + sales_other)  (product concentration)
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Return numerator/denominator, or None if either is None or denominator is ~zero."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-9:
        return None
    return round(numerator / denominator, 4)


def _revenue(sales_main: Optional[float], sales_other: Optional[float]) -> Optional[float]:
    if sales_main is None and sales_other is None:
        return None
    return (sales_main or 0.0) + (sales_other or 0.0)


@tool
def compute_financial_health(
    sales_main: Optional[float],
    sales_other: Optional[float],
    monthly_budget: Optional[float],
    investments: Optional[float],
) -> dict:
    """
    Compute derived financial metrics for a company.

    Parameters
    ----------
    sales_main      : Primary revenue stream (e.g. product/service sales).
    sales_other     : Secondary revenue stream (e.g. ancillary income).
    monthly_budget  : Stated monthly operating budget.
    investments     : Total capital invested in the business.

    Returns
    -------
    dict with keys:
        revenue_estimate         : Total revenue (sales_main + sales_other)
        budget_utilisation_ratio : revenue / monthly_budget; >1 means revenue covers costs
        revenue_per_investment   : revenue / investments; capital efficiency
        sales_mix_ratio          : sales_main / total_revenue; product concentration (1 = pure main)
        data_quality_issues      : List of missing/zero fields that affected computation
    """
    issues: list[str] = []

    # Track missing inputs
    if sales_main is None:
        issues.append("missing: sales_main")
    if sales_other is None:
        issues.append("missing: sales_other")
    if monthly_budget is None:
        issues.append("missing: monthly_budget")
    elif abs(monthly_budget) < 1e-9:
        issues.append("zero: monthly_budget")
    if investments is None:
        issues.append("missing: investments")
    elif abs(investments) < 1e-9:
        issues.append("zero: investments")

    revenue = _revenue(sales_main, sales_other)

    budget_utilisation_ratio = _safe_ratio(revenue, monthly_budget)
    revenue_per_investment = _safe_ratio(revenue, investments)

    # sales_mix_ratio: how concentrated the revenue is in the primary channel
    if sales_main is None or revenue is None or abs(revenue) < 1e-9:
        sales_mix_ratio = None
        if revenue is not None and abs(revenue) < 1e-9:
            issues.append("zero: total_revenue (sales_main + sales_other)")
    else:
        sales_mix_ratio = round(sales_main / revenue, 4)

    result = {
        "revenue_estimate": round(revenue, 2) if revenue is not None else None,
        "budget_utilisation_ratio": budget_utilisation_ratio,
        "revenue_per_investment": revenue_per_investment,
        "sales_mix_ratio": sales_mix_ratio,
        "data_quality_issues": issues,
    }

    if issues:
        logger.debug("compute_financial_health: data quality issues — %s", issues)

    return result
