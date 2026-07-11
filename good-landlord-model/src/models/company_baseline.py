"""
company_baseline.py — Out-of-fold company baseline + adjusted landlord score.

Pipeline
--------
1. Train a LogisticRegression on CompanyIsActive using StratifiedKFold OOF so
   each company's predicted probability comes from a model that never saw it.
2. Compute residual = actual − predicted_prob for each company.
   Positive residual → company outperformed what the market predicts.
   Negative residual → company underperformed.
3. Aggregate residuals per landlord (mean across their portfolio).
4. Apply Empirical-Bayes shrinkage to correct for landlords with small portfolios:
       AdjustedScore = N/(N+m) * RawScore + m/(N+m) * GlobalMean
   where m = cfg["shrinkage_m"] (default 10).

Public API
----------
compute_oof_predictions(model_base, feature_cols, cfg)
    -> pd.DataFrame with added columns:
         OOFPredictedProb  float  [0, 1]
         OOFResidual       float  actual - predicted
         OOFFold           int    fold index (-1 = excluded)

aggregate_landlord_scores(model_base_with_oof, bridge, cfg)
    -> pd.DataFrame (landlord-level):
         LandLordID, TenantCount, RawAdjustedScore, AdjustedScore

build_landlord_adjusted_scores(model_base, bridge, feature_cols, cfg)
    -> (model_base_with_oof, landlord_scores)  — convenience wrapper
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_DEFAULT_N_SPLITS: int = 5
_DEFAULT_SHRINKAGE_M: float = 10.0
_DEFAULT_SEED: int = 42

# Columns excluded from auto-detected feature set
_EXCLUDE_COLS: frozenset[str] = frozenset({
    "CompanyID",
    "LandLordID",
    "CompanyIsActive",
    "SuccessScore",
    "SuccessScoreRank",
    "OOFPredictedProb",
    "OOFResidual",
    "OOFFold",
    "CompanyStatus",
    "PrimaryType",
    "OriginCity",
    "OriginCountry",
    "PreferredIndustry",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cfg(cfg: dict | None) -> dict:
    """
    Accept either the full training config or a subsection.

    Tries keys: 'baseline', 'model'; falls back to root dict.
    """
    if cfg is None:
        return {}
    for key in ("baseline", "model"):
        if key in cfg and isinstance(cfg[key], dict):
            return cfg[key]
    return cfg


def _auto_feature_cols(df: pd.DataFrame, exclude: frozenset = _EXCLUDE_COLS) -> list[str]:
    """Return numeric columns that are not in the exclude set."""
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def _make_pipeline(seed: int, C: float = 1.0) -> Pipeline:
    """Median-impute → StandardScaler → L2 LogisticRegression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            max_iter=1_000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# OOF predictions
# ---------------------------------------------------------------------------


def compute_oof_predictions(
    model_base: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg: dict | None = None,
    *,
    return_model: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, Pipeline]":
    """
    Fit LogisticRegression OOF on CompanyIsActive and append predictions.

    Each row's predicted probability comes from a model fold that excluded it,
    preventing label leakage into the company-level residuals.

    Parameters
    ----------
    model_base    : DataFrame containing 'CompanyIsActive' + numeric features.
                    Rows where CompanyIsActive is NaN are skipped (OOF cols → NaN).
    feature_cols  : explicit feature list; None = auto-detect numeric columns
    cfg           : training config dict; keys used:
                      seed         (int, default 42)
                      n_splits     (int, default 5)
                      lr_C         (float, default 1.0)
    return_model  : if True, return (DataFrame, Pipeline) where the Pipeline
                    is fitted on ALL labeled rows (for production scoring)

    Returns
    -------
    pd.DataFrame with original columns plus:
        OOFPredictedProb  float   predicted P(active) from held-out fold
        OOFResidual       float   CompanyIsActive − OOFPredictedProb
        OOFFold           int     fold index; -1 for excluded (NaN-target) rows
    """
    if "CompanyIsActive" not in model_base.columns:
        raise ValueError(
            "'CompanyIsActive' column not found in model_base. "
            "Run build_targets() before compute_oof_predictions()."
        )

    resolved = _resolve_cfg(cfg)
    seed = int(resolved.get("seed", _DEFAULT_SEED))
    n_splits = int(resolved.get("n_splits", _DEFAULT_N_SPLITS))
    lr_C = float(resolved.get("lr_C", 1.0))

    # Separate labeled / unlabeled rows
    labeled_mask = model_base["CompanyIsActive"].notna()
    n_excluded = int((~labeled_mask).sum())
    if n_excluded:
        logger.warning(
            "%d rows with NaN CompanyIsActive excluded from OOF training.", n_excluded
        )

    df_labeled = model_base[labeled_mask].copy()
    y = df_labeled["CompanyIsActive"].astype(float).values

    # Feature selection
    if feature_cols is None:
        feature_cols = _auto_feature_cols(df_labeled)

    missing_cols = [c for c in feature_cols if c not in df_labeled.columns]
    if missing_cols:
        raise ValueError(
            f"feature_cols not found in model_base: {missing_cols}"
        )

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns available for LogisticRegression. "
            "Pass feature_cols explicitly or add numeric columns to model_base."
        )

    X = df_labeled[feature_cols].values.astype(float)
    n_labeled = len(df_labeled)

    logger.info(
        "OOF LR: n_labeled=%d, n_features=%d, n_splits=%d, seed=%d, C=%.3f",
        n_labeled, len(feature_cols), n_splits, seed, lr_C,
    )

    # Clamp n_splits if minority class is too small
    class_counts = np.bincount(y.astype(int))
    min_class = int(class_counts.min()) if len(class_counts) >= 2 else 0
    if min_class == 0:
        raise ValueError(
            "CompanyIsActive has only one class — cannot train a classifier. "
            "Check the binary_target output."
        )
    if min_class < n_splits:
        n_splits = max(2, min_class)
        logger.warning(
            "Clamped n_splits to %d (minority class size = %d).", n_splits, min_class
        )

    pipe = _make_pipeline(seed=seed, C=lr_C)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.full(n_labeled, np.nan)
    oof_folds = np.full(n_labeled, -1, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
        oof_folds[val_idx] = fold_idx
        logger.debug(
            "Fold %d: val_n=%d, mean_prob=%.3f",
            fold_idx, len(val_idx), oof_probs[val_idx].mean(),
        )

    # Fit final model on all labeled data
    pipe.fit(X, y)

    # Write back into a copy of the full model_base
    out = model_base.copy()
    out["OOFPredictedProb"] = np.nan
    out["OOFResidual"] = np.nan
    out["OOFFold"] = -1

    idx = df_labeled.index
    out.loc[idx, "OOFPredictedProb"] = oof_probs
    out.loc[idx, "OOFResidual"] = y - oof_probs
    out.loc[idx, "OOFFold"] = oof_folds

    logger.info(
        "OOF complete: mean_prob=%.3f, mean_residual=%.4f, "
        "residual_std=%.4f",
        float(np.nanmean(oof_probs)),
        float(np.nanmean(y - oof_probs)),
        float(np.nanstd(y - oof_probs)),
    )

    if return_model:
        return out, pipe
    return out


# ---------------------------------------------------------------------------
# Landlord-level aggregation + Empirical-Bayes shrinkage
# ---------------------------------------------------------------------------


def aggregate_landlord_scores(
    model_base: pd.DataFrame,
    bridge: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-company residuals to the landlord level with EB shrinkage.

    Empirical-Bayes formula
    -----------------------
    GlobalMean  = mean(OOFResidual) across all companies with valid predictions
    For each landlord with N companies and mean residual μ_N:
        AdjustedScore = (N / (N + m)) * μ_N  +  (m / (N + m)) * GlobalMean

    A landlord with many tenants (N >> m) keeps their raw mean.
    A landlord with few tenants (N << m) shrinks toward the global mean.

    Parameters
    ----------
    model_base : DataFrame output of compute_oof_predictions; must contain
                 'LandLordID' and 'OOFResidual'.
    bridge     : (LandLordID, CompanyID) bridge — currently unused in the
                 calculation but kept for signature compatibility and future
                 use (e.g., portfolio-size-based features).
    cfg        : training config; key used:
                   shrinkage_m  (float, default 10.0)

    Returns
    -------
    pd.DataFrame with columns (sorted descending by AdjustedScore):
        LandLordID        str
        TenantCount       int    companies with valid OOF predictions
        RawAdjustedScore  float  unshrunken mean residual
        AdjustedScore     float  EB-shrunk score
    """
    if "OOFResidual" not in model_base.columns:
        raise ValueError(
            "'OOFResidual' not found. Run compute_oof_predictions() first."
        )
    if "LandLordID" not in model_base.columns:
        raise ValueError(
            "'LandLordID' not found in model_base."
        )

    resolved = _resolve_cfg(cfg)
    m = float(resolved.get("shrinkage_m", _DEFAULT_SHRINKAGE_M))

    valid = model_base[model_base["OOFResidual"].notna()].copy()
    if valid.empty:
        logger.warning("No valid OOF residuals — returning empty landlord scores.")
        return pd.DataFrame(
            columns=["LandLordID", "TenantCount", "RawAdjustedScore", "AdjustedScore"]
        )

    global_mean = float(valid["OOFResidual"].mean())

    agg = (
        valid.groupby("LandLordID", sort=False)["OOFResidual"]
        .agg(TenantCount="count", RawAdjustedScore="mean")
        .reset_index()
    )

    logger.info(
        "Landlord aggregation: %d landlords, global_mean_residual=%.4f, m=%.1f",
        len(agg), global_mean, m,
    )

    n = agg["TenantCount"].astype(float)
    if m == 0:
        # No shrinkage
        agg["AdjustedScore"] = agg["RawAdjustedScore"]
    else:
        agg["AdjustedScore"] = (
            (n / (n + m)) * agg["RawAdjustedScore"]
            + (m / (n + m)) * global_mean
        )

    agg = agg.sort_values("AdjustedScore", ascending=False).reset_index(drop=True)

    logger.info(
        "AdjustedScore: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        float(agg["AdjustedScore"].mean()),
        float(agg["AdjustedScore"].std()),
        float(agg["AdjustedScore"].min()),
        float(agg["AdjustedScore"].max()),
    )
    return agg


# ---------------------------------------------------------------------------
# Convenience end-to-end wrapper
# ---------------------------------------------------------------------------


def build_landlord_adjusted_scores(
    model_base: pd.DataFrame,
    bridge: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: OOF predictions → residuals → landlord EB-shrunk scores.

    Parameters
    ----------
    model_base   : bridge LEFT JOIN companies (output of build_model_base)
    bridge       : (LandLordID, CompanyID) bridge
    feature_cols : feature column names for the LR; None = auto-detect numerics
    cfg          : training config

    Returns
    -------
    (model_base_with_oof, landlord_scores)
        model_base_with_oof : input model_base + OOFPredictedProb + OOFResidual + OOFFold
        landlord_scores     : landlord-level DataFrame with AdjustedScore
    """
    model_base_with_oof = compute_oof_predictions(
        model_base, feature_cols=feature_cols, cfg=cfg
    )
    landlord_scores = aggregate_landlord_scores(
        model_base_with_oof, bridge, cfg=cfg
    )
    return model_base_with_oof, landlord_scores
