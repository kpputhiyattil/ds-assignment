"""
train.py — Multi-model landlord regression with GroupKFold cross-validation.

Supported model types
---------------------
  catboost     CatBoostRegressor — native categorical support
  xgboost      XGBRegressor      — ordinal-encoded categoricals
  lightgbm     LGBMRegressor     — ordinal-encoded categoricals
  random_forest RandomForestRegressor — ordinal-encoded categoricals

Cross-validation
----------------
GroupKFold(n_splits=cv_folds) grouped by LandLordID prevents any landlord
from leaking across train/validation folds.  For the landlord feature matrix
each landlord appears exactly once, so GroupKFold ≡ KFold — but using
GroupKFold keeps the API consistent with the company-baseline model and
future multi-row-per-landlord extensions.

Public API
----------
CVResult                               dataclass — per-fold + aggregate results
train_cv(matrix, numeric_cols,
         categorical_cols, model_type,
         cfg, n_splits)                -> CVResult
train_all(matrix, numeric_cols,
          categorical_cols, cfg,
          n_splits)                    -> dict[str, CVResult]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

ModelType = Literal["catboost", "xgboost", "lightgbm", "random_forest"]
ALL_MODEL_TYPES: list[ModelType] = ["catboost", "xgboost", "lightgbm", "random_forest"]

_TARGET_COL = "AdjustedScore"
_GROUP_COL  = "LandLordID"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    mae: float
    rmse: float
    r2: float
    spearman: float
    n_val: int


@dataclass
class CVResult:
    """Full cross-validation result for one model type."""
    model_type: str
    oof_predictions: np.ndarray    # shape (n_landlords,)
    oof_true: np.ndarray           # shape (n_landlords,)
    landlord_ids: np.ndarray       # shape (n_landlords,)
    fold_results: list[FoldResult]
    final_model: Any               # trained on all data
    feature_importance: pd.DataFrame  # columns: feature, importance

    @property
    def mean_metrics(self) -> dict[str, float]:
        return {
            "mae":      float(np.mean([f.mae      for f in self.fold_results])),
            "rmse":     float(np.mean([f.rmse     for f in self.fold_results])),
            "r2":       float(np.mean([f.r2       for f in self.fold_results])),
            "spearman": float(np.mean([f.spearman for f in self.fold_results])),
        }

    @property
    def std_metrics(self) -> dict[str, float]:
        return {
            "mae":      float(np.std([f.mae      for f in self.fold_results])),
            "rmse":     float(np.std([f.rmse     for f in self.fold_results])),
            "r2":       float(np.std([f.r2       for f in self.fold_results])),
            "spearman": float(np.std([f.spearman for f in self.fold_results])),
        }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_model_cfg(cfg: dict | None, model_type: str) -> dict:
    """
    Extract model-specific config section from the full training config.
    Falls back to empty dict (use defaults) if section is absent.
    """
    if cfg is None:
        return {}
    # Support both full config and landlord_model subsection
    lm_cfg = cfg.get("landlord_model", cfg)
    return lm_cfg.get(model_type, {})


def _resolve_cv_folds(cfg: dict | None) -> int:
    if cfg is None:
        return 5
    lm_cfg = cfg.get("landlord_model", cfg)
    return int(lm_cfg.get("cv_folds", 5))


# ---------------------------------------------------------------------------
# Categorical encoding for non-CatBoost models
# ---------------------------------------------------------------------------

class _CatEncoder:
    """Fit/transform object-typed columns with OrdinalEncoder."""

    def __init__(self) -> None:
        self._enc: OrdinalEncoder | None = None
        self._cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
        self._cols = cat_cols
        if not cat_cols:
            return np.empty((len(df), 0), dtype=float)
        X = df[cat_cols].fillna("__missing__").astype(str)
        self._enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        return self._enc.fit_transform(X).astype(float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._cols:
            return np.empty((len(df), 0), dtype=float)
        X = df[self._cols].fillna("__missing__").astype(str)
        return self._enc.transform(X).astype(float)  # type: ignore[union-attr]


class _NumImputer:
    """Median-impute numeric columns."""

    def __init__(self) -> None:
        self._imp: SimpleImputer | None = None
        self._cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame, num_cols: list[str]) -> np.ndarray:
        self._cols = num_cols
        if not num_cols:
            return np.empty((len(df), 0), dtype=float)
        self._imp = SimpleImputer(strategy="median")
        return self._imp.fit_transform(df[num_cols].astype(float))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._cols:
            return np.empty((len(df), 0), dtype=float)
        return self._imp.transform(df[self._cols].astype(float))  # type: ignore[union-attr]


def _encode_sklearn(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Encode train/val DataFrames into numpy arrays for sklearn-compatible models.

    Returns (X_tr, X_vl, all_feature_names).
    """
    num_enc = _NumImputer()
    cat_enc = _CatEncoder()

    num_tr = num_enc.fit_transform(X_train, numeric_cols)
    num_vl = num_enc.transform(X_val)

    cat_tr = cat_enc.fit_transform(X_train, cat_cols)
    cat_vl = cat_enc.transform(X_val)

    X_tr = np.hstack([num_tr, cat_tr]) if (num_tr.size or cat_tr.size) else np.empty((len(X_train), 0))
    X_vl = np.hstack([num_vl, cat_vl]) if (num_vl.size or cat_vl.size) else np.empty((len(X_val), 0))
    feat_names = numeric_cols + cat_cols
    return X_tr, X_vl, feat_names


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_catboost(model_cfg: dict, cat_col_indices: list[int], seed: int):
    from catboost import CatBoostRegressor
    params = dict(
        iterations=model_cfg.get("iterations", 1000),
        depth=model_cfg.get("depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.03),
        loss_function=model_cfg.get("loss_function", "RMSE"),
        eval_metric=model_cfg.get("eval_metric", "MAE"),
        l2_leaf_reg=model_cfg.get("l2_leaf_reg", 5),
        random_seed=model_cfg.get("random_seed", seed),
        verbose=model_cfg.get("verbose", False),
        allow_writing_files=False,
    )
    return CatBoostRegressor(**params), cat_col_indices


def _build_xgboost(model_cfg: dict, seed: int):
    from xgboost import XGBRegressor
    params = dict(
        n_estimators=model_cfg.get("n_estimators", 1000),
        max_depth=model_cfg.get("max_depth", 5),
        learning_rate=model_cfg.get("learning_rate", 0.03),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        random_state=model_cfg.get("random_state", seed),
        n_jobs=-1,
        verbosity=0,
    )
    early = model_cfg.get("early_stopping_rounds")
    if early is not None:
        params["early_stopping_rounds"] = int(early)
    return XGBRegressor(**params)


def _build_lightgbm(model_cfg: dict, seed: int):
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators=model_cfg.get("n_estimators", 1000),
        max_depth=model_cfg.get("max_depth", 5),
        learning_rate=model_cfg.get("learning_rate", 0.03),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        random_state=model_cfg.get("random_state", seed),
        n_jobs=-1,
        verbose=model_cfg.get("verbose", -1),
    )
    return LGBMRegressor(**params)


def _build_random_forest(model_cfg: dict, seed: int):
    params = dict(
        n_estimators=model_cfg.get("n_estimators", 500),
        max_depth=model_cfg.get("max_depth", 8),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 2),
        random_state=model_cfg.get("random_state", seed),
        n_jobs=-1,
    )
    return RandomForestRegressor(**params)


# ---------------------------------------------------------------------------
# Fold-level metric computation
# ---------------------------------------------------------------------------

def _fold_metrics(fold: int, y_true: np.ndarray, y_pred: np.ndarray) -> FoldResult:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    corr = spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else float("nan")
    spearman = float(corr) if corr is not None and not np.isnan(corr) else float("nan")
    return FoldResult(fold=fold, mae=mae, rmse=rmse, r2=r2, spearman=spearman, n_val=len(y_true))


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _extract_importance(
    model: Any,
    feature_names: list[str],
    model_type: str,
    cat_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return DataFrame(feature, importance) sorted descending."""
    try:
        if model_type == "catboost":
            importances = model.get_feature_importance()
        else:
            importances = model.feature_importances_
        importances = np.array(importances, dtype=float)
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as exc:
        logger.warning("Could not extract feature importance for %s: %s", model_type, exc)
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})


# ---------------------------------------------------------------------------
# Core CV training function
# ---------------------------------------------------------------------------

def train_cv(
    matrix: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_type: ModelType = "catboost",
    cfg: dict | None = None,
    n_splits: int | None = None,
) -> CVResult:
    """
    Train one model type with GroupKFold cross-validation.

    Parameters
    ----------
    matrix          : output of build_feature_matrix — one row per landlord
    numeric_cols    : numeric feature column names
    categorical_cols: categorical (object dtype) feature column names
    model_type      : one of ALL_MODEL_TYPES
    cfg             : full training config dict (or None for defaults)
    n_splits        : override cv_folds from cfg

    Returns
    -------
    CVResult with OOF predictions, per-fold metrics, and final model.
    """
    if model_type not in ALL_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {ALL_MODEL_TYPES}, got {model_type!r}")

    if _TARGET_COL not in matrix.columns:
        raise ValueError(f"matrix must contain '{_TARGET_COL}' column.")
    if _GROUP_COL not in matrix.columns:
        raise ValueError(f"matrix must contain '{_GROUP_COL}' column for GroupKFold.")

    seed       = int((cfg or {}).get("seed", 42))
    model_cfg  = _resolve_model_cfg(cfg, model_type)
    cv_folds   = n_splits if n_splits is not None else _resolve_cv_folds(cfg)

    # Clamp n_splits to number of rows (can't have more folds than samples)
    n_rows  = len(matrix)
    n_splits_actual = max(2, min(cv_folds, n_rows))
    if n_splits_actual != cv_folds:
        logger.warning(
            "Clamping cv_folds from %d to %d (only %d landlords in matrix).",
            cv_folds, n_splits_actual, n_rows,
        )

    feat_cols   = numeric_cols + categorical_cols
    y           = matrix[_TARGET_COL].values.astype(float)
    groups      = matrix[_GROUP_COL].values
    X_df        = matrix[feat_cols]

    oof_pred    = np.full(n_rows, np.nan)
    fold_results: list[FoldResult] = []

    gkf = GroupKFold(n_splits=n_splits_actual)

    logger.info(
        "train_cv [%s]: %d landlords, %d features (%d num + %d cat), %d folds",
        model_type, n_rows, len(feat_cols), len(numeric_cols), len(categorical_cols),
        n_splits_actual,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_df, y, groups)):
        X_train_df = X_df.iloc[train_idx]
        X_val_df   = X_df.iloc[val_idx]
        y_train    = y[train_idx]
        y_val      = y[val_idx]

        if model_type == "catboost":
            from catboost import Pool
            cat_idx = list(range(len(numeric_cols), len(feat_cols)))
            model, _ = _build_catboost(model_cfg, cat_idx, seed)
            early = model_cfg.get("early_stopping_rounds", 100)

            # CatBoost: pass DataFrame directly with cat_features as column indices
            X_tr_cb = X_train_df.copy()
            X_vl_cb = X_val_df.copy()
            # Fill NaN in categoricals with placeholder for CatBoost
            for col in categorical_cols:
                X_tr_cb[col] = X_tr_cb[col].fillna("__missing__")
                X_vl_cb[col] = X_vl_cb[col].fillna("__missing__")

            train_pool = Pool(X_tr_cb, y_train, cat_features=categorical_cols)
            val_pool   = Pool(X_vl_cb, y_val,   cat_features=categorical_cols)

            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early,
            )
            y_pred = model.predict(val_pool)
            feat_names_cv = feat_cols

        else:
            X_tr, X_vl, feat_names_cv = _encode_sklearn(
                X_train_df, X_val_df, numeric_cols, categorical_cols
            )

            if model_type == "xgboost":
                early = model_cfg.get("early_stopping_rounds", 50)
                model = _build_xgboost(model_cfg, seed)
                model.fit(
                    X_tr, y_train,
                    eval_set=[(X_vl, y_val)],
                    verbose=False,
                )
                y_pred = model.predict(X_vl)

            elif model_type == "lightgbm":
                import lightgbm as lgb
                early = model_cfg.get("early_stopping_rounds", 50)
                model = _build_lightgbm(model_cfg, seed)
                callbacks = [lgb.early_stopping(early, verbose=False), lgb.log_evaluation(-1)]
                model.fit(
                    X_tr, y_train,
                    eval_set=[(X_vl, y_val)],
                    callbacks=callbacks,
                )
                y_pred = model.predict(X_vl)

            elif model_type == "random_forest":
                model = _build_random_forest(model_cfg, seed)
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_vl)

            else:
                raise ValueError(f"Unhandled model_type: {model_type}")

        oof_pred[val_idx] = y_pred
        fold_results.append(_fold_metrics(fold_idx, y_val, y_pred))
        logger.debug(
            "  Fold %d/%d  MAE=%.4f  RMSE=%.4f  R²=%.4f  Spearman=%.4f",
            fold_idx + 1, n_splits_actual,
            fold_results[-1].mae, fold_results[-1].rmse,
            fold_results[-1].r2, fold_results[-1].spearman,
        )

    # ── Final model trained on all data ──────────────────────────────────────
    logger.info("  Training final %s on all %d landlords …", model_type, n_rows)

    if model_type == "catboost":
        from catboost import Pool
        X_all = X_df.copy()
        for col in categorical_cols:
            X_all[col] = X_all[col].fillna("__missing__")
        final_model, _ = _build_catboost(model_cfg, list(range(len(numeric_cols), len(feat_cols))), seed)
        # Disable early stopping for the final model (no eval set)
        final_model.set_params(early_stopping_rounds=None)
        final_pool = Pool(X_all, y, cat_features=categorical_cols)
        final_model.fit(final_pool)
        feat_names_final = feat_cols

    else:
        num_enc_final = _NumImputer()
        cat_enc_final = _CatEncoder()
        num_all = num_enc_final.fit_transform(X_df, numeric_cols)
        cat_all = cat_enc_final.fit_transform(X_df, categorical_cols)
        X_all_np = np.hstack([num_all, cat_all]) if (num_all.size or cat_all.size) else np.empty((n_rows, 0))
        feat_names_final = numeric_cols + categorical_cols

        if model_type == "xgboost":
            final_model = _build_xgboost(model_cfg, seed)
            # No eval set on full-data fit — disable early stopping
            final_model.set_params(early_stopping_rounds=None)
            final_model.fit(X_all_np, y, verbose=False)
        elif model_type == "lightgbm":
            final_model = _build_lightgbm(model_cfg, seed)
            final_model.fit(X_all_np, y)
        else:  # random_forest
            final_model = _build_random_forest(model_cfg, seed)
            final_model.fit(X_all_np, y)

    importance_df = _extract_importance(final_model, feat_names_final, model_type, categorical_cols)

    mean_m = {k: round(v, 5) for k, v in {
        "mae": np.mean([f.mae for f in fold_results]),
        "rmse": np.mean([f.rmse for f in fold_results]),
        "r2": np.mean([f.r2 for f in fold_results]),
        "spearman": np.mean([f.spearman for f in fold_results]),
    }.items()}
    logger.info(
        "train_cv [%s] OOF mean → MAE=%.4f RMSE=%.4f R²=%.4f Spearman=%.4f",
        model_type, mean_m["mae"], mean_m["rmse"], mean_m["r2"], mean_m["spearman"],
    )

    return CVResult(
        model_type=model_type,
        oof_predictions=oof_pred,
        oof_true=y,
        landlord_ids=groups,
        fold_results=fold_results,
        final_model=final_model,
        feature_importance=importance_df,
    )


# ---------------------------------------------------------------------------
# Train all model types
# ---------------------------------------------------------------------------

def train_all(
    matrix: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: dict | None = None,
    n_splits: int | None = None,
    model_types: list[ModelType] | None = None,
) -> dict[str, CVResult]:
    """
    Train all supported model types and return results keyed by model name.

    Parameters
    ----------
    matrix          : output of build_feature_matrix
    numeric_cols    : numeric feature column names
    categorical_cols: categorical feature column names
    cfg             : full training config dict
    n_splits        : override cv_folds for all models
    model_types     : subset to train (defaults to ALL_MODEL_TYPES)

    Returns
    -------
    dict mapping model_type → CVResult
    """
    types_to_run = model_types if model_types is not None else ALL_MODEL_TYPES
    results: dict[str, CVResult] = {}
    for mtype in types_to_run:
        logger.info("=== Training: %s ===", mtype)
        results[mtype] = train_cv(
            matrix, numeric_cols, categorical_cols,
            model_type=mtype, cfg=cfg, n_splits=n_splits,
        )
    return results
