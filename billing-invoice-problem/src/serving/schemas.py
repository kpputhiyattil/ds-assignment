"""Pydantic request/response schemas for the decision API (Step 8)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    date: dt.date
    amount: float = Field(gt=0)


class AssessRequest(BaseModel):
    customer_id: str
    invoices: list[Invoice] = Field(min_length=1)
    snapshot: dt.date | None = None
    explain: bool = True


class AssessResponse(BaseModel):
    customer_id: str
    snapshot: str
    recommendation: str
    reason: str
    detail: str
    auto_decided: bool
    churn_probability: float | None = None
    expected_dollar_churn: float | None = None
    inferred_interval: str | None = None
    interval_confidence: float | None = None
    interval_reason: str | None = None
    decision_reason: str | None = None
    with_interval: dict[str, Any] | None = None
    without_interval: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    model_version: str | None = None
    briefing: dict | None = None


class BatchSummary(BaseModel):
    n_customers: int
    recommendation_counts: dict[str, int]
    mean_churn_probability: float
    total_expected_dollar_churn: float
    mean_interval_confidence: float
    n_explained: int
    recommendation_counts_without_interval: dict[str, int] | None = None
    agreement_rate: float | None = None


class BatchAssessResponse(BaseModel):
    snapshot: str
    n_customers: int
    summary: BatchSummary
    results: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    comparison_summary: dict[str, Any] | None = None
    model_version: str | None = None
    source_filename: str | None = None
    truncated: bool = False
    max_customers_applied: int | None = None
    n_invoice_rows_loaded: int | None = None
    warning: str | None = None
