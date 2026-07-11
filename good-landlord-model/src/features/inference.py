"""
inference.py — Build landlord feature matrices for scoring (same transforms as training).

Training uses ``build_feature_matrix`` (requires AdjustedScore). Serving uses
``prepare_upload_for_scoring`` / ``build_scoring_feature_matrix`` which apply the
same ``landlord_features`` + ``portfolio_features`` pipeline without a training
target. These helpers are packed into ``LandlordScorer`` (transform /
score_end_to_end).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.features.transforms import landlord_features, portfolio_features

logger = logging.getLogger(__name__)

InputKind = Literal["feature_matrix", "raw_landlords", "landlord_ids", "unknown"]

_RAW_MARKERS = frozenset({
    "AllCompanyID",
    "YearFounded",
    "TotalPopulationAround",
    "MonthlyBudget",  # company-side; rare on landlord file
})
_FEATURE_MARKERS = frozenset({
    "LandlordAge",
    "PopulationDensity",
    "PortfolioMeanClients",
    "PortfolioActiveRate",
    "PortfolioMeanBudget",
    "AreaPerCompany",
})


def detect_input_kind(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> InputKind:
    """
    Classify an uploaded table so the scorer can route it correctly.
    """
    cols = set(df.columns)
    num = numeric_cols or []
    cat = categorical_cols or []
    needed = num + cat
    if needed:
        hit = sum(1 for c in needed if c in cols)
        if hit >= max(6, int(0.5 * len(needed))):
            return "feature_matrix"

    feat_hits = len(cols & _FEATURE_MARKERS)
    raw_hits = len(cols & _RAW_MARKERS)
    if raw_hits >= 1 and feat_hits < 3:
        return "raw_landlords"
    if cols <= {"LandLordID", "TenantCount"} and "LandLordID" in cols:
        return "landlord_ids"
    if feat_hits >= 3:
        return "feature_matrix"
    if "LandLordID" in cols and raw_hits >= 1:
        return "raw_landlords"
    return "unknown"


def attach_company_targets(
    model_base: pd.DataFrame,
    companies_scored: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CompanyIsActive / SuccessScore onto model_base (same as training runner)."""
    target_cols = [
        c for c in ("CompanyID", "CompanyIsActive", "SuccessScore", "SuccessScoreRank")
        if c in companies_scored.columns
    ]
    drop_existing = [c for c in target_cols if c != "CompanyID" and c in model_base.columns]
    base = model_base.drop(columns=drop_existing, errors="ignore")
    return base.merge(companies_scored[target_cols], on="CompanyID", how="left")


def build_scoring_feature_matrix(
    landlords: pd.DataFrame,
    *,
    companies: pd.DataFrame,
    companies_scored: pd.DataFrame | None = None,
    cfg: dict | None = None,
    feature_set: Literal["landlord_only", "with_portfolio"] = "with_portfolio",
) -> pd.DataFrame:
    """
    Apply the training feature-engineering path for inference.

    Steps (mirrors ``scripts/run_landlord_models.py`` stages 1–2 + 5, without
    company-baseline AdjustedScore which is the *target*, not a feature):
      1. build_model_base(landlords, companies)
      2. attach company success targets when available
      3. landlord_features(landlords)
      4. portfolio_features(model_base) when feature_set=with_portfolio
      5. left-join portfolio onto landlord features
    """
    from src.data.ingestion import build_model_base

    if "LandLordID" not in landlords.columns:
        raise ValueError("Landlords table must include LandLordID.")

    bridge, model_base = build_model_base(landlords, companies)
    if companies_scored is not None and len(companies_scored):
        model_base = attach_company_targets(model_base, companies_scored)

    ll_feats = landlord_features(landlords, cfg=cfg)
    if feature_set == "with_portfolio":
        pf_feats = portfolio_features(model_base, cfg=cfg)
        matrix = ll_feats.merge(pf_feats, on="LandLordID", how="left")
    else:
        matrix = ll_feats

    if "TenantCount" not in matrix.columns and "PortfolioSize" in matrix.columns:
        matrix = matrix.copy()
        matrix["TenantCount"] = matrix["PortfolioSize"]

    logger.info(
        "build_scoring_feature_matrix (%s): %d landlords, %d columns "
        "(bridge rows=%d)",
        feature_set, len(matrix), matrix.shape[1], len(bridge),
    )
    return matrix.reset_index(drop=True)


def _ensure_feature_columns(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_cols:
        if c not in out.columns:
            out[c] = np.nan
    for c in categorical_cols:
        if c not in out.columns:
            out[c] = None
    return out


def _default_companies() -> pd.DataFrame:
    from src.data.ingestion import load_companies

    return load_companies()


def _default_companies_scored(cfg: dict | None = None) -> pd.DataFrame | None:
    from src.config import cfg as default_cfg
    from src.targets.construction import build_targets

    use_cfg = cfg or default_cfg
    processed = Path(use_cfg["paths"]["processed_dir"])
    targets_path = processed / "company_targets.parquet"
    if targets_path.exists():
        logger.info("Using cached company targets: %s", targets_path)
        return pd.read_parquet(targets_path)

    logger.info("Building company targets for scoring pipeline …")
    companies = _default_companies()
    scored = build_targets(companies, cfg=use_cfg)
    processed.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(targets_path, index=False)
    return scored


def _default_landlords() -> pd.DataFrame:
    from src.data.ingestion import load_landlords

    return load_landlords()


def prepare_upload_for_scoring(
    df: pd.DataFrame,
    *,
    companies: pd.DataFrame | None = None,
    companies_scored: pd.DataFrame | None = None,
    landlords_lookup: pd.DataFrame | None = None,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    feature_set: Literal["landlord_only", "with_portfolio"] = "with_portfolio",
    cfg: dict | None = None,
    force_pipeline: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Route an upload through the same FE used at training time.

    Packed into ``LandlordScorer.transform`` / ``score_end_to_end``.

    Returns
    -------
    (feature_matrix, pipeline_info)
    """
    num = list(numeric_cols or [])
    cat = list(categorical_cols or [])
    kind = detect_input_kind(df, numeric_cols=num, categorical_cols=cat)

    info: dict[str, Any] = {
        "input_kind": kind,
        "feature_set": feature_set,
        "pipeline_applied": False,
        "n_input_rows": int(len(df)),
    }

    if kind == "feature_matrix" and not force_pipeline:
        info["note"] = (
            "Upload already looks like a model feature matrix; "
            "scoring directly without re-running feature engineering."
        )
        return _ensure_feature_columns(df, num, cat), info

    landlords = df
    if kind == "landlord_ids":
        if "LandLordID" not in df.columns:
            raise ValueError("Upload must include LandLordID.")
        ref = landlords_lookup if landlords_lookup is not None else _default_landlords()
        ids = set(df["LandLordID"].astype(str))
        landlords = ref[ref["LandLordID"].astype(str).isin(ids)].copy()
        if landlords.empty:
            raise ValueError(
                "None of the uploaded LandLordIDs were found in the project LandLords data."
            )
        info["n_matched_landlords"] = int(len(landlords))
    elif kind in {"raw_landlords", "unknown"} or force_pipeline:
        if "LandLordID" not in df.columns:
            raise ValueError(
                "Raw landlord upload must include LandLordID "
                "(and typically AllCompanyID / YearFounded / Area, …)."
            )
        if kind == "unknown" and "AllCompanyID" not in df.columns and not force_pipeline:
            info["note"] = (
                "Could not detect raw landlord schema; scoring with available columns. "
                "Upload raw LandLords rows (with AllCompanyID) or the feature matrix."
            )
            return _ensure_feature_columns(df, num, cat), info
        landlords = df

    if companies is None:
        companies = _default_companies()
    if companies_scored is None:
        companies_scored = _default_companies_scored(cfg)

    matrix = build_scoring_feature_matrix(
        landlords,
        companies=companies,
        companies_scored=companies_scored,
        cfg=cfg,
        feature_set=feature_set,
    )
    info["pipeline_applied"] = True
    info["n_feature_rows"] = int(len(matrix))
    info["note"] = (
        "Ran training-aligned feature engineering: "
        "build_model_base -> company targets -> landlord_features -> portfolio_features."
    )
    return _ensure_feature_columns(matrix, num, cat), info
