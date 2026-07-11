"""
construction.py — Company-level success target construction.

Two target definitions are produced for every company:

CompanyIsActive  (binary, int)
    1 = company currently Active, 0 = Closed / Suspended / other.
    Used as the label for the company-baseline LogisticRegression that
    estimates expected performance before landlord adjustment.

SuccessScore  (continuous, float in [0, 1])
    Composite of five rank-normalised performance signals:
      sales_efficiency  = SalesOfMainProduct / (MonthlyBudget + eps)
      retention_rate    = ReturningClient / (TotalActiveClients + eps)
      conversion_rate   = ClientsInTheLast6Month / (TotalVisitorsInTheLast6Month + eps)
      momentum          = (ClientsInTheLast12Month - ClientsInTheLast6Month)
                          / (TotalVisitorsInTheLast12Month + eps)
      client_scale      = TotalActiveClients  (absolute size signal)

    Each signal is rank-normalised to [0, 1], then combined via weighted sum
    (weights from configs/training_config.yaml -> composite_weights).

Public API
----------
binary_target(companies, ...)       -> pd.Series[int]
rank_normalize(series)              -> pd.Series[float]
composite_score(companies, ...)     -> pd.Series[float]
build_targets(companies, cfg)       -> pd.DataFrame  (companies + target cols)
sensitivity_analysis(companies, ..) -> pd.DataFrame  (rank-corr across weight grids)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small epsilon prevents division by zero in ratio features
_EPS = 1e-6


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as float Series; if absent, return NaN Series of same length."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


# Default active labels (case-insensitive match)
_DEFAULT_ACTIVE_LABELS: frozenset[str] = frozenset({"active"})


# ---------------------------------------------------------------------------
# Binary target
# ---------------------------------------------------------------------------


def binary_target(
    companies: pd.DataFrame,
    status_col: str = "CompanyStatus",
    active_labels: set[str] | None = None,
) -> pd.Series:
    """
    Map CompanyStatus to a binary 0/1 target.

    Parameters
    ----------
    companies     : DataFrame with at least `status_col`
    status_col    : column containing status strings
    active_labels : set of values treated as "active" (case-insensitive);
                    defaults to {"active"}

    Returns
    -------
    pd.Series[int] named 'CompanyIsActive', same index as `companies`.
    Missing values in `status_col` -> 0 (conservatively treated as inactive).
    """
    labels = {lbl.lower() for lbl in (active_labels or _DEFAULT_ACTIVE_LABELS)}

    if status_col not in companies.columns:
        raise ValueError(
            f"Column '{status_col}' not found in companies DataFrame. "
            f"Available: {list(companies.columns)}"
        )

    raw = companies[status_col].fillna("").astype(str).str.strip().str.lower()
    target = raw.isin(labels).astype(int)
    target.name = "CompanyIsActive"

    n_active = target.sum()
    logger.info(
        "Binary target: %d active / %d total (%.1f%%)",
        n_active, len(target), n_active / len(target) * 100,
    )
    return target


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    Percentile-rank normalise a numeric series to [0, 1].

    NaN values remain NaN. Ties receive the average rank.
    Result is (rank - 1) / (n_valid - 1): min -> 0, max -> 1.
    """
    valid = series.notna()
    n = valid.sum()
    if n == 0:
        return series.copy().astype(float)
    if n == 1:
        out = series.copy().astype(float)
        out[valid] = 0.5
        return out

    ranked = series.rank(method="average", na_option="keep")
    normalised = (ranked - 1) / (n - 1)
    normalised.name = series.name
    return normalised


# ---------------------------------------------------------------------------
# Component extractors
# ---------------------------------------------------------------------------


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / (denominator + eps), clipped to [0, 10] to cap outliers."""
    return (numerator / (denominator + _EPS)).clip(lower=0, upper=10)


def _sales_efficiency(df: pd.DataFrame) -> pd.Series:
    """Revenue per unit of budget — proxy for ROI."""
    return _safe_ratio(_get_col(df, "SalesOfMainProduct"), _get_col(df, "MonthlyBudget")).rename("sales_efficiency")


def _retention_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of active clients who are returning — loyalty signal."""
    return _safe_ratio(_get_col(df, "ReturningClient"), _get_col(df, "TotalActiveClients")).rename("retention_rate")


def _conversion_rate(df: pd.DataFrame) -> pd.Series:
    """6-month clients / 6-month visitors — footfall-to-client efficiency."""
    return _safe_ratio(
        _get_col(df, "ClientsInTheLast6Month"),
        _get_col(df, "TotalVisitorsInTheLast6Month"),
    ).rename("conversion_rate")


def _momentum(df: pd.DataFrame) -> pd.Series:
    """
    Net new clients in the most-recent 6-month window, normalised by 12-month visitors.
    Positive = growing; clipped at 0 to guard against data artefacts.
    """
    net_new = (_get_col(df, "ClientsInTheLast12Month") - _get_col(df, "ClientsInTheLast6Month")).clip(lower=0)
    return _safe_ratio(net_new, _get_col(df, "TotalVisitorsInTheLast12Month")).rename("momentum")


def _client_scale(df: pd.DataFrame) -> pd.Series:
    """Absolute active client count — scale / reach signal."""
    return _get_col(df, "TotalActiveClients").clip(lower=0).rename("client_scale")


_COMPONENT_EXTRACTORS = {
    "sales_efficiency": _sales_efficiency,
    "retention":        _retention_rate,
    "conversion":       _conversion_rate,
    "momentum":         _momentum,
    "active_clients":   _client_scale,
}

# Default weights (must sum to 1.0; overridden by cfg["composite_weights"])
_DEFAULT_WEIGHTS: dict[str, float] = {
    "sales_efficiency": 0.30,
    "retention":        0.20,
    "conversion":       0.20,
    "momentum":         0.15,
    "active_clients":   0.15,
}


# ---------------------------------------------------------------------------
# Composite SuccessScore
# ---------------------------------------------------------------------------


def composite_score(
    companies: pd.DataFrame,
    weights: dict[str, float] | None = None,
    return_components: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.DataFrame]":
    """
    Compute a composite SuccessScore in [0, 1] for each company.

    Steps: extract raw signals -> rank-normalise each to [0,1] -> weighted sum.

    Parameters
    ----------
    companies         : Companies DataFrame
    weights           : component name -> weight (must sum ~1); None = default
    return_components : if True, also return normalised-component DataFrame

    Returns
    -------
    pd.Series[float] named 'SuccessScore', or (score, comp_df) if return_components.
    """
    w = weights or _DEFAULT_WEIGHTS.copy()

    w_total = sum(w.values())
    if abs(w_total - 1.0) > 1e-6:
        logger.warning("Composite weights sum to %.4f — normalising to 1.0", w_total)
        w = {k: v / w_total for k, v in w.items()}

    components: dict[str, pd.Series] = {}
    for name, extractor in _COMPONENT_EXTRACTORS.items():
        if name not in w:
            continue
        raw_component = extractor(companies)
        raw_component.index = companies.index
        components[name] = rank_normalize(raw_component)

    comp_df = pd.DataFrame(components, index=companies.index)

    score = pd.Series(0.0, index=companies.index, name="SuccessScore")
    weight_used = pd.Series(0.0, index=companies.index)

    for name, normalised in components.items():
        weight = w.get(name, 0.0)
        valid = normalised.notna()
        score[valid] += normalised[valid] * weight
        weight_used[valid] += weight

    # Rescale partial scores (some components missing)
    partial = (weight_used > 0) & (weight_used < 1 - 1e-6)
    if partial.any():
        score[partial] = score[partial] / weight_used[partial]

    # All components missing -> NaN
    score[weight_used == 0] = np.nan

    logger.info(
        "SuccessScore: mean=%.3f median=%.3f std=%.3f (n=%d, n_nan=%d)",
        score.mean(), score.median(), score.std(),
        score.notna().sum(), score.isna().sum(),
    )

    if return_components:
        return score, comp_df
    return score


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


def build_targets(
    companies: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Add CompanyIsActive, SuccessScore, and SuccessScoreRank to companies.

    Parameters
    ----------
    companies : raw or cleaned Companies DataFrame
    cfg       : config dict (src.config.cfg); if None, uses defaults

    Returns
    -------
    pd.DataFrame with added columns:
        CompanyIsActive  int    0 / 1
        SuccessScore     float  0.0 - 1.0
        SuccessScoreRank int    1 = best (dense rank)
    """
    weights = None
    if cfg is not None:
        weights = cfg.get("composite_weights")

    out = companies.copy()
    out["CompanyIsActive"] = binary_target(out)
    out["SuccessScore"] = composite_score(out, weights=weights)
    out["SuccessScoreRank"] = (
        out["SuccessScore"]
        .rank(method="dense", ascending=False, na_option="bottom")
        .astype(int)
    )

    logger.info(
        "build_targets: %d companies — %d active (%.0f%%), SuccessScore [%.3f, %.3f]",
        len(out), out["CompanyIsActive"].sum(), out["CompanyIsActive"].mean() * 100,
        out["SuccessScore"].min(), out["SuccessScore"].max(),
    )
    return out


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def sensitivity_analysis(
    companies: pd.DataFrame,
    weight_configs: list[dict[str, float]] | None = None,
    config_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score companies under multiple weight configs; return Spearman corr matrix.

    Values near 1.0 = the two configs produce nearly identical rankings.
    Low values = rankings are sensitive to that weight choice.
    """
    if weight_configs is None:
        weight_configs = [
            _DEFAULT_WEIGHTS,
            {"sales_efficiency": 0.50, "retention": 0.15, "conversion": 0.15,
             "momentum": 0.10, "active_clients": 0.10},
            {"sales_efficiency": 0.15, "retention": 0.45, "conversion": 0.20,
             "momentum": 0.10, "active_clients": 0.10},
            {"sales_efficiency": 0.10, "retention": 0.15, "conversion": 0.30,
             "momentum": 0.35, "active_clients": 0.10},
            {"sales_efficiency": 0.20, "retention": 0.20, "conversion": 0.20,
             "momentum": 0.20, "active_clients": 0.20},
        ]

    if config_names is None:
        config_names = [
            "default", "sales_heavy", "retention_heavy", "growth_focused", "equal"
        ][:len(weight_configs)]

    scores = {name: composite_score(companies, weights=w)
              for name, w in zip(config_names, weight_configs)}

    corr = pd.DataFrame(scores, index=companies.index).corr(method="spearman").round(4)

    logger.info(
        "Sensitivity: min Spearman corr=%.3f",
        corr.values[~np.eye(len(corr), dtype=bool)].min() if len(corr) > 1 else 1.0,
    )
    return corr
