"""
transforms.py — Landlord feature engineering.

Two feature sets are produced:

  landlord_only
      Features derivable from LandLords.parquet alone.
      Safe to use for brand-new landlords with no tenant history.

  with_portfolio
      landlord_only  +  portfolio-level aggregations from the company
      model_base (output of build_model_base / build_targets).
      Richer signal but requires at least one matched company.

Pipeline summary
----------------
1. landlord_features(landlords)   → per-landlord numeric + categorical features
2. portfolio_features(model_base) → per-landlord tenant-portfolio aggregations
3. build_feature_matrix(...)      → join + target attach; ready for model training

Public API
----------
landlord_features(landlords, cfg)                    -> pd.DataFrame  [LandLordID + features]
portfolio_features(model_base, cfg)                  -> pd.DataFrame  [LandLordID + features]
build_feature_matrix(landlords, model_base,
                     landlord_scores, cfg,
                     feature_set)                    -> pd.DataFrame  [LandLordID + X + y]
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Column-name aliases — handle alternative column names in raw data
# ---------------------------------------------------------------------------

# Some LandLords extracts use prefixed names; normalise on load.
_LANDLORD_COL_ALIASES: dict[str, str] = {
    "LandlordOriginCity":    "OriginCity",
    "LandlordOriginCountry": "OriginCountry",
}

# Categorical columns forwarded as-is (CatBoost handles them natively)
_LANDLORD_CAT_COLS: list[str] = ["PreferredIndustry", "OriginCity", "OriginCountry"]

# Numeric columns taken directly from LandLords (no transformation)
_LANDLORD_NUM_RAW: list[str] = ["Area", "TotalPopulationAround", "ActiveCompanies"]

# Company columns used for portfolio aggregations
_PORTFOLIO_AGG_COLS: list[str] = [
    "MonthlyBudget",
    "SalesOfMainProduct",
    "TotalActiveClients",
    "ReturningClient",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as float Series; NaN Series of correct length if absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / (denominator + eps); result is NaN where numerator is NaN."""
    result = numerator / (denominator.abs() + _EPS)
    result[numerator.isna()] = np.nan
    return result


def _normalise_landlord_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column-name aliases in place (copy)."""
    rename = {src: dst for src, dst in _LANDLORD_COL_ALIASES.items() if src in df.columns}
    if rename:
        df = df.rename(columns=rename)
        logger.debug("Renamed landlord columns: %s", rename)
    return df


# ---------------------------------------------------------------------------
# 1.  Landlord-level features
# ---------------------------------------------------------------------------


def landlord_features(
    landlords: pd.DataFrame,
    cfg: dict | None = None,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """
    Derive per-landlord features from the LandLords DataFrame.

    Numeric features
    ----------------
    LandlordAge          int/float  years since YearFounded (NaN if unknown)
    Area                 float      raw area value
    TotalPopulationAround float     raw population
    ActiveCompanies      float      claimed tenant count
    PopulationDensity    float      TotalPopulationAround / (Area + eps)
    AreaPerCompany       float      Area / (ActiveCompanies + eps)
    PopulationPerCompany float      TotalPopulationAround / (ActiveCompanies + eps)

    Categorical features (string; forwarded unchanged for CatBoost)
    ---------------------------------------------------------------
    PreferredIndustry    str | None
    OriginCity           str | None
    OriginCountry        str | None

    Parameters
    ----------
    landlords      : raw LandLords DataFrame (must contain LandLordID)
    cfg            : training config (unused currently; reserved for future thresholds)
    reference_year : year to compute LandlordAge against; defaults to current year

    Returns
    -------
    pd.DataFrame indexed 0..n-1 with LandLordID as the first column.
    """
    if "LandLordID" not in landlords.columns:
        raise ValueError("'LandLordID' column not found in landlords DataFrame.")

    df = _normalise_landlord_cols(landlords.copy())

    if reference_year is None:
        reference_year = pd.Timestamp.now().year

    out = pd.DataFrame({"LandLordID": df["LandLordID"].astype(str)})

    # ── Age ──────────────────────────────────────────────────────────────────
    year_founded = _get_num(df, "YearFounded")
    landlord_age = reference_year - year_founded
    # Clamp implausible values (negative age or >300 years)
    landlord_age = landlord_age.where(
        (landlord_age >= 0) & (landlord_age <= 300), other=np.nan
    )
    out["LandlordAge"] = landlord_age

    # ── Raw numerics ─────────────────────────────────────────────────────────
    area        = _get_num(df, "Area").clip(lower=0)
    population  = _get_num(df, "TotalPopulationAround").clip(lower=0)
    n_companies = _get_num(df, "ActiveCompanies").clip(lower=0)

    out["Area"]                 = area
    out["TotalPopulationAround"] = population
    out["ActiveCompanies"]      = n_companies

    # ── Ratio features ───────────────────────────────────────────────────────
    out["PopulationDensity"]    = _safe_ratio(population, area)
    out["AreaPerCompany"]       = _safe_ratio(area,       n_companies)
    out["PopulationPerCompany"] = _safe_ratio(population, n_companies)

    # ── Categorical pass-through ─────────────────────────────────────────────
    for col in _LANDLORD_CAT_COLS:
        if col in df.columns:
            # Convert to string, map blank / "nan" → None
            s = df[col].astype(str).str.strip()
            s = s.replace({"": None, "nan": None, "None": None, "NaN": None})
            out[col] = s
        else:
            out[col] = None

    n_numeric_feats = 7          # age + 3 raw + 3 ratios
    n_cat_feats     = len(_LANDLORD_CAT_COLS)
    logger.info(
        "landlord_features: %d landlords, %d numeric + %d categorical features",
        len(out), n_numeric_feats, n_cat_feats,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.  Portfolio features
# ---------------------------------------------------------------------------


def portfolio_features(
    model_base: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Aggregate company-level signals to the landlord level.

    Input is the model_base DataFrame (bridge LEFT JOIN companies), optionally
    enriched with Step-4 target columns (CompanyIsActive, SuccessScore) and
    Step-5 OOF columns (OOFResidual).

    Output features (one row per LandLordID)
    ----------------------------------------
    PortfolioSize         int    number of matched companies
    PortfolioActiveRate   float  fraction with CompanyIsActive == 1
    PortfolioMeanBudget   float  mean MonthlyBudget
    PortfolioStdBudget    float  std MonthlyBudget (NaN if only 1 company)
    PortfolioMeanSales    float  mean SalesOfMainProduct
    PortfolioMeanClients  float  mean TotalActiveClients
    PortfolioMeanRetention float mean ReturningClient / (TotalActiveClients + eps)
    PortfolioMeanSuccessScore float  mean SuccessScore (NaN if column absent)
    PortfolioStdSuccessScore  float  std SuccessScore  (NaN if column absent)

    Parameters
    ----------
    model_base : DataFrame with LandLordID + company columns
    cfg        : training config (unused currently; reserved)

    Returns
    -------
    pd.DataFrame with LandLordID as first column.
    """
    if "LandLordID" not in model_base.columns:
        raise ValueError("'LandLordID' column not found in model_base.")

    agg_rows: dict[str, list] = {
        "LandLordID":              [],
        "PortfolioSize":           [],
        "PortfolioActiveRate":     [],
        "PortfolioMeanBudget":     [],
        "PortfolioStdBudget":      [],
        "PortfolioMeanSales":      [],
        "PortfolioMeanClients":    [],
        "PortfolioMeanRetention":  [],
        "PortfolioMeanSuccessScore": [],
        "PortfolioStdSuccessScore":  [],
    }

    has_active  = "CompanyIsActive" in model_base.columns
    has_score   = "SuccessScore"    in model_base.columns
    has_returning = "ReturningClient" in model_base.columns and "TotalActiveClients" in model_base.columns

    for landlord_id, grp in model_base.groupby("LandLordID", sort=False):
        agg_rows["LandLordID"].append(str(landlord_id))
        agg_rows["PortfolioSize"].append(len(grp))

        # Active rate
        if has_active:
            valid_active = grp["CompanyIsActive"].dropna()
            agg_rows["PortfolioActiveRate"].append(
                float(valid_active.mean()) if len(valid_active) > 0 else np.nan
            )
        else:
            agg_rows["PortfolioActiveRate"].append(np.nan)

        # Budget
        budget = _get_num(grp, "MonthlyBudget")
        agg_rows["PortfolioMeanBudget"].append(float(budget.mean()) if budget.notna().any() else np.nan)
        agg_rows["PortfolioStdBudget"].append(float(budget.std())  if budget.notna().sum() > 1 else np.nan)

        # Sales
        sales = _get_num(grp, "SalesOfMainProduct")
        agg_rows["PortfolioMeanSales"].append(float(sales.mean()) if sales.notna().any() else np.nan)

        # Clients
        clients = _get_num(grp, "TotalActiveClients")
        agg_rows["PortfolioMeanClients"].append(float(clients.mean()) if clients.notna().any() else np.nan)

        # Retention rate per company (within group)
        if has_returning:
            returning = _get_num(grp, "ReturningClient")
            ret_rate  = _safe_ratio(returning, clients)
            agg_rows["PortfolioMeanRetention"].append(
                float(ret_rate.mean()) if ret_rate.notna().any() else np.nan
            )
        else:
            agg_rows["PortfolioMeanRetention"].append(np.nan)

        # SuccessScore
        if has_score:
            score = _get_num(grp, "SuccessScore")
            agg_rows["PortfolioMeanSuccessScore"].append(float(score.mean()) if score.notna().any() else np.nan)
            agg_rows["PortfolioStdSuccessScore"].append(float(score.std())   if score.notna().sum() > 1 else np.nan)
        else:
            agg_rows["PortfolioMeanSuccessScore"].append(np.nan)
            agg_rows["PortfolioStdSuccessScore"].append(np.nan)

    result = pd.DataFrame(agg_rows)
    logger.info(
        "portfolio_features: %d landlords, %d feature columns",
        len(result), result.shape[1] - 1,
    )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.  Full feature matrix
# ---------------------------------------------------------------------------


_FEATURE_SET_LANDLORD_ONLY = "landlord_only"
_FEATURE_SET_WITH_PORTFOLIO = "with_portfolio"
_VALID_FEATURE_SETS = {_FEATURE_SET_LANDLORD_ONLY, _FEATURE_SET_WITH_PORTFOLIO}

# Columns that are not features (identifiers / target)
_NON_FEATURE_COLS: frozenset[str] = frozenset({
    "LandLordID", "AdjustedScore", "RawAdjustedScore", "TenantCount",
})


def build_feature_matrix(
    landlords: pd.DataFrame,
    model_base: pd.DataFrame,
    landlord_scores: pd.DataFrame,
    cfg: dict | None = None,
    feature_set: Literal["landlord_only", "with_portfolio"] = "with_portfolio",
) -> pd.DataFrame:
    """
    Assemble the final feature matrix ready for landlord model training.

    Steps
    -----
    1. Compute landlord_features from LandLords DataFrame.
    2. Optionally compute portfolio_features from model_base (if feature_set="with_portfolio").
    3. Merge in the AdjustedScore target from landlord_scores.
    4. Inner join on LandLordID so only scored landlords are included.

    Parameters
    ----------
    landlords        : raw LandLords DataFrame
    model_base       : bridge LEFT JOIN companies (+target cols from build_targets)
    landlord_scores  : output of aggregate_landlord_scores — must contain
                       LandLordID and AdjustedScore
    cfg              : training config
    feature_set      : 'landlord_only' | 'with_portfolio'

    Returns
    -------
    pd.DataFrame with columns:
        LandLordID       str
        <feature cols>   numeric + categorical
        AdjustedScore    float   (the regression target)
        TenantCount      int     (kept for weighting / filtering)
    """
    if feature_set not in _VALID_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {_VALID_FEATURE_SETS}, got {feature_set!r}"
        )
    if "AdjustedScore" not in landlord_scores.columns:
        raise ValueError("landlord_scores must contain 'AdjustedScore'.")
    if "LandLordID" not in landlord_scores.columns:
        raise ValueError("landlord_scores must contain 'LandLordID'.")

    # 1. Landlord features
    ll_feats = landlord_features(landlords, cfg=cfg)

    # 2. Portfolio features (optional)
    if feature_set == _FEATURE_SET_WITH_PORTFOLIO:
        pf_feats = portfolio_features(model_base, cfg=cfg)
        matrix   = ll_feats.merge(pf_feats, on="LandLordID", how="left")
    else:
        matrix = ll_feats

    # 3. Merge target
    target_cols = ["LandLordID", "AdjustedScore"]
    if "TenantCount" in landlord_scores.columns:
        target_cols.append("TenantCount")

    matrix = matrix.merge(
        landlord_scores[target_cols],
        on="LandLordID",
        how="inner",     # only keep landlords that have a score
    )

    n_rows = len(matrix)
    feat_cols = [c for c in matrix.columns if c not in _NON_FEATURE_COLS]
    n_feats   = len(feat_cols)
    logger.info(
        "build_feature_matrix (%s): %d landlords, %d feature columns, "
        "target mean=%.4f",
        feature_set, n_rows, n_feats,
        float(matrix["AdjustedScore"].mean()) if n_rows > 0 else float("nan"),
    )
    return matrix.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utility: feature column lists for downstream model training
# ---------------------------------------------------------------------------


def get_feature_cols(
    matrix: pd.DataFrame,
    feature_set: str = "with_portfolio",
) -> tuple[list[str], list[str]]:
    """
    Return (numeric_cols, categorical_cols) for the feature matrix.

    Excludes identifier and target columns so the lists can be passed
    directly to CatBoost or sklearn pipelines.

    Parameters
    ----------
    matrix      : output of build_feature_matrix
    feature_set : used only for logging

    Returns
    -------
    (numeric_cols, categorical_cols)
    """
    exclude = _NON_FEATURE_COLS
    numeric_cols     = [c for c in matrix.select_dtypes(include="number").columns if c not in exclude]
    categorical_cols = [c for c in matrix.select_dtypes(include="object").columns  if c not in exclude]
    logger.info(
        "get_feature_cols (%s): %d numeric, %d categorical",
        feature_set, len(numeric_cols), len(categorical_cols),
    )
    return numeric_cols, categorical_cols
