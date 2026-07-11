"""Pydantic request/response schemas for the /assess API (Step 8)."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class Invoice(BaseModel):
    date: dt.date
    amount: float = Field(gt=0)


class AssessRequest(BaseModel):
    customer_id: str
    invoices: list[Invoice] = Field(min_length=1)
    snapshot: dt.date | None = None  # defaults to today


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
    model_version: str | None = None
    briefing: dict | None = None
