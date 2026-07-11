"""
Pydantic request / response schemas for the scoring API.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LandlordRecord(BaseModel):
    """Single landlord feature row (keys match training feature matrix)."""

    LandLordID: str | None = None
    TenantCount: float | None = None
    LandlordAge: float | None = None
    Area: float | None = None
    TotalPopulationAround: float | None = None
    ActiveCompanies: float | None = None
    PopulationDensity: float | None = None
    AreaPerCompany: float | None = None
    PopulationPerCompany: float | None = None
    PreferredIndustry: str | None = None
    OriginCity: str | None = None
    OriginCountry: str | None = None
    PortfolioSize: float | None = None
    PortfolioActiveRate: float | None = None
    PortfolioMeanBudget: float | None = None
    PortfolioStdBudget: float | None = None
    PortfolioMeanSales: float | None = None
    PortfolioMeanClients: float | None = None
    PortfolioMeanRetention: float | None = None
    PortfolioMeanSuccessScore: float | None = None
    PortfolioStdSuccessScore: float | None = None

    model_config = {"extra": "allow"}


class ScoreRequest(BaseModel):
    landlords: list[LandlordRecord] = Field(
        ...,
        min_length=1,
        description="One or more landlord feature rows",
    )
    top_k: int = Field(5, ge=1, le=15)


class ScoreResponse(BaseModel):
    n_input: int
    n_scored: int
    truncated: bool
    max_rows: int
    model_type: str | None = None
    model_version: str | None = None
    results: list[dict[str, Any]]
