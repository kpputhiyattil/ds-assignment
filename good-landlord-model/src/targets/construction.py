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
      conversion_rate   = ClientsInTheLast6Months / (TotalClientsInTheLast6Months + eps)
      momentum          = (ClientsInTheLast12Months - ClientsInTheLast6Months)
                          / (TotalClientsInTheLast12Months + eps)
      client_scale      = TotalActiveClients  (absolute size signal)

    Each signal is rank-normalised to [0, 1], then combined via weighted sum
    (weights from configs/training_config.yaml -> target.composite_weights).

Public API
----------
binary_target(companies, ...)       -> pd.Series
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


def _get_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first matching column as float; NaN Series if none exist."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


# Default active labels (case-insensitive match) — overridden by config
_DEFAULT_ACTIVE_LABELS: frozenset[str] = frozenset({"active"})


# ---------------------------------------------------------------------------
# Binary target
# ---------------------------------------------------------------------------


def binary_target(
    companies: pd.DataFrame,
    status_col: str = "CompanyStatus",
    active_labels: set[str] | list[str] | None = None,
    inactive_labels: set[str] | list[str] | None = None,
    exclude_labels: set[str] | list[str] | None = None,
) -> pd.Series:
    """
    Map CompanyStatus to a binary 0/1 target (NaN = excluded / unknown).

    Parameters
    ----------
    companies       : DataFrame with at least `status_col`
    status_col      : column containing status strings
    active_labels   : treated as 1 (case-insensitive); default {"active"}
    inactive_labels : treated as 0; if None, everything not active is 0 (legacy)
    exclude_labels  : treated as NaN (dropped from binary training)

    Returns
    -------
    pd.Series named 'CompanyIsActive', same index as `companies`.
    """
    labels_pos = {lbl.lower() for lbl in (active_labels or _DEFAULT_ACTIVE_LABELS)}
    labels_neg = (
        {lbl.lower() for lbl in inactive_labels} if inactive_labels is not None else None
    )
    labels_ex = (
        {lbl.lower() for lbl in exclude_labels} if exclude_labels is not None else set()
    )

    if status_col not in companies.columns:
        raise ValueError(
            f"Column '{status_col}' not found in companies DataFrame. "
            f"Available: {list(companies.columns)}"
        )

    raw = companies[status_col].fillna("").astype(str).str.strip().str.lower()
    missing_mask = raw.isin({"", "none", "nan", "null"}) | companies[status_col].isna()
    raw = raw.mask(missing_mask, other=pd.NA)

    target = pd.Series(np.nan, index=companies.index, dtype=float)
    is_pos = raw.isin(labels_pos)
    target[is_pos] = 1.0

    if labels_neg is not None:
        target[raw.isin(labels_neg)] = 0.0
    else:
        # Legacy: non-active (including missing) -> 0
        target[:] = 0.0
        target[is_pos] = 1.0

    if labels_ex:
        target[raw.isin(labels_ex)] = np.nan

    target.name = "CompanyIsActive"

    n_active = int((target == 1).sum())
    n_inactive = int((target == 0).sum())
    n_excl = int(target.isna().sum())
    logger.info(
        "Binary target: %d active / %d inactive / %d excluded-or-missing (total=%d)",
        n_active, n_inactive, n_excl, len(target),
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
    return _safe_ratio(
        _get_col(df, "SalesOfMainProduct"),
        _get_col(df, "MonthlyBudget", "Monthly budget"),
    ).rename("sales_efficiency")


def _retention_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of active clients who are returning — loyalty signal."""
    return _safe_ratio(
        _get_col(df, "ReturningClient"),
        _get_col(df, "TotalActiveClients"),
    ).rename("retention_rate")


def _conversion_rate(df: pd.DataFrame) -> pd.Series:
    """6-month clients / 6-month totals — engagement efficiency."""
    return _safe_ratio(
        _get_col(df, "ClientsInTheLast6Months", "ClientsInTheLast6Month"),
        _get_col(
            df,
            "TotalClientsInTheLast6Months",
            "TotalVisitorsInTheLast6Month",
            "TotalClientsInTheLast6Month",
        ),
    ).rename("conversion_rate")


def _momentum(df: pd.DataFrame) -> pd.Series:
    """
    Net new clients in the most-recent 6-month window, normalised by 12-month totals.
    Positive = growing; clipped at 0 to guard against data artefacts.
    """
    clients_12 = _get_col(df, "ClientsInTheLast12Months", "ClientsInTheLast12Month")
    clients_6 = _get_col(df, "ClientsInTheLast6Months", "ClientsInTheLast6Month")
    denom = _get_col(
        df,
        "TotalClientsInTheLast12Months",
        "TotalVisitorsInTheLast12Month",
        "TotalClientsInTheLast12Month",
    )
    net_new = (clients_12 - clients_6).clip(lower=0)
    return _safe_ratio(net_new, denom).rename("momentum")


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

# Default weights (must sum to 1.0; overridden by cfg["target"]["composite_weights"])
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


def _target_section(cfg: dict | None) -> dict:
    """Accept full training cfg or a bare target/weights dict."""
    if cfg is None:
        return {}
    if "target" in cfg and isinstance(cfg["target"], dict):
        return cfg["target"]
    return cfg


def build_targets(
    companies: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Add CompanyIsActive, SuccessScore, and SuccessScoreRank to companies.

    Parameters
    ----------
    companies : raw or cleaned Companies DataFrame
    cfg       : full training config (src.config.cfg) or a target subsection;
                if None, uses defaults

    Returns
    -------
    pd.DataFrame with added columns:
        CompanyIsActive  float  0 / 1 / NaN
        SuccessScore     float  0.0 - 1.0
        SuccessScoreRank int    1 = best (dense rank)
    """
    target_cfg = _target_section(cfg)
    weights = target_cfg.get("composite_weights")

    out = companies.copy()
    out["CompanyIsActive"] = binary_target(
        out,
        active_labels=target_cfg.get("status_positive"),
        inactive_labels=target_cfg.get("status_negative"),
        exclude_labels=target_cfg.get("status_exclude"),
    )
    out["SuccessScore"] = composite_score(out, weights=weights)
    out["SuccessScoreRank"] = (
        out["SuccessScore"]
        .rank(method="dense", ascending=False, na_option="bottom")
        .astype(int)
    )

    n_labeled = int(out["CompanyIsActive"].notna().sum())
    n_active = int((out["CompanyIsActive"] == 1).sum())
    logger.info(
        "build_targets: %d companies — %d labeled binary (%d active, %.0f%% of labeled), "
        "SuccessScore [%.3f, %.3f]",
        len(out),
        n_labeled,
        n_active,
        (n_active / n_labeled * 100) if n_labeled else 0.0,
        out["SuccessScore"].min(),
        out["SuccessScore"].max(),
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
