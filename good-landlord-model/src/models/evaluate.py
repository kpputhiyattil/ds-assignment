"""
evaluate.py — Evaluation metrics and analysis for landlord regression models.

Public API
----------
regression_metrics(y_true, y_pred)       -> dict   MAE, RMSE, R², Spearman
lift_table(y_true, y_pred, n_bins)       -> pd.DataFrame
quality_band_metrics(y_true, y_pred,
                     top_pct, bot_pct)   -> dict
compare_models(results)                  -> pd.DataFrame
suggest_best_model(comparison)           -> dict
render_model_comparison_report(...)      -> str
oof_residual_summary(result)             -> pd.DataFrame
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train import CVResult

logger = logging.getLogger(__name__)


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table without requiring the tabulate package."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.5f}" if abs(v) < 1e3 else f"{v:.4g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


# ---------------------------------------------------------------------------
# Core regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute MAE, RMSE, R², and Spearman rank correlation.

    NaN pairs are removed before computation.

    Parameters
    ----------
    y_true  : ground-truth AdjustedScore values
    y_pred  : model predictions
    prefix  : optional string prefix for metric keys (e.g. "oof_")

    Returns
    -------
    dict with keys: {prefix}mae, {prefix}rmse, {prefix}r2,
                    {prefix}spearman, {prefix}n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = int(mask.sum())
    if n == 0:
        return {f"{prefix}mae": np.nan, f"{prefix}rmse": np.nan,
                f"{prefix}r2": np.nan, f"{prefix}spearman": np.nan,
                f"{prefix}n": 0}

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if n > 1 else np.nan
    spearman_result = spearmanr(y_true, y_pred) if n > 1 else None
    spearman = float(spearman_result.statistic) if spearman_result is not None else np.nan

    return {
        f"{prefix}mae":      mae,
        f"{prefix}rmse":     rmse,
        f"{prefix}r2":       r2,
        f"{prefix}spearman": spearman,
        f"{prefix}n":        n,
    }


# ---------------------------------------------------------------------------
# Lift table
# ---------------------------------------------------------------------------

def lift_table(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute a lift table by sorting landlords on predicted score.

    The table ranks landlords by their predicted AdjustedScore (descending),
    bins them into ``n_bins`` equal-size groups, and reports the mean
    predicted and actual score per bin — exposing whether the model
    correctly separates good from bad landlords.

    Parameters
    ----------
    y_true  : true AdjustedScore values
    y_pred  : model OOF (or test) predictions
    n_bins  : number of equal-size bins (deciles when n_bins=10)

    Returns
    -------
    pd.DataFrame with columns:
        bin          int   (1 = highest predicted, n_bins = lowest)
        n            int   number of landlords in bin
        mean_pred    float mean predicted score
        mean_actual  float mean actual AdjustedScore
        lift         float mean_actual / global_mean_actual
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    n = len(y_true)
    if n == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "mean_actual", "lift"])

    n_bins = max(1, min(n_bins, n))
    global_mean = float(np.mean(y_true))

    # Sort descending by predicted score (best first → bin 1)
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    bin_edges = np.array_split(np.arange(n), n_bins)
    rows = []
    for bin_num, idx in enumerate(bin_edges, start=1):
        mean_pred   = float(np.mean(y_pred_sorted[idx]))
        mean_actual = float(np.mean(y_true_sorted[idx]))
        lift        = (mean_actual / global_mean) if abs(global_mean) > 1e-9 else np.nan
        rows.append({
            "bin":         bin_num,
            "n":           len(idx),
            "mean_pred":   mean_pred,
            "mean_actual": mean_actual,
            "lift":        lift,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality-band accuracy
# ---------------------------------------------------------------------------

def quality_band_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    top_percentile: float = 0.70,
    bottom_percentile: float = 0.30,
) -> dict[str, float]:
    """
    Measure how well predicted rankings capture the true top/bottom landlords.

    Defines "true top" as landlords with AdjustedScore >= top_percentile
    quantile of y_true (and similarly for bottom).  Then checks what fraction
    of the predicted top / bottom matches the true top / bottom.

    Parameters
    ----------
    y_true           : true AdjustedScore values
    y_pred           : predicted scores
    top_percentile   : upper quantile threshold (0-1), default 0.70 (top 30%)
    bottom_percentile: lower quantile threshold (0-1), default 0.30 (bottom 30%)

    Returns
    -------
    dict with keys:
        top_capture_rate    float  fraction of true-top correctly predicted in top
        bottom_capture_rate float  fraction of true-bottom correctly in predicted bottom
        n_true_top          int
        n_true_bottom       int
        n_pred_top          int
        n_pred_bottom       int
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {k: np.nan for k in (
            "top_capture_rate", "bottom_capture_rate",
            "n_true_top", "n_true_bottom", "n_pred_top", "n_pred_bottom"
        )}

    top_thr_true    = float(np.quantile(y_true, top_percentile))
    bottom_thr_true = float(np.quantile(y_true, bottom_percentile))
    top_thr_pred    = float(np.quantile(y_pred, top_percentile))
    bottom_thr_pred = float(np.quantile(y_pred, bottom_percentile))

    true_top    = y_true >= top_thr_true
    true_bottom = y_true <= bottom_thr_true
    pred_top    = y_pred >= top_thr_pred
    pred_bottom = y_pred <= bottom_thr_pred

    n_true_top    = int(true_top.sum())
    n_true_bottom = int(true_bottom.sum())
    n_pred_top    = int(pred_top.sum())
    n_pred_bottom = int(pred_bottom.sum())

    top_capture    = float((true_top & pred_top).sum()    / n_true_top)    if n_true_top    > 0 else np.nan
    bottom_capture = float((true_bottom & pred_bottom).sum() / n_true_bottom) if n_true_bottom > 0 else np.nan

    return {
        "top_capture_rate":    top_capture,
        "bottom_capture_rate": bottom_capture,
        "n_true_top":          n_true_top,
        "n_true_bottom":       n_true_bottom,
        "n_pred_top":          n_pred_top,
        "n_pred_bottom":       n_pred_bottom,
    }


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, "CVResult"]) -> pd.DataFrame:
    """
    Summarise OOF metrics for all trained models in a single DataFrame.

    Parameters
    ----------
    results : dict mapping model_type → CVResult (output of train_all)

    Returns
    -------
    pd.DataFrame with one row per model and columns:
        model_type, mae_mean, mae_std, rmse_mean, rmse_std,
        r2_mean, r2_std, spearman_mean, spearman_std,
        n_folds, n_landlords
    Sorted ascending by mae_mean (best model first).
    """
    rows = []
    for model_type, result in results.items():
        m  = result.mean_metrics
        s  = result.std_metrics
        row = {
            "model_type":     model_type,
            "mae_mean":       round(m["mae"],      5),
            "mae_std":        round(s["mae"],      5),
            "rmse_mean":      round(m["rmse"],     5),
            "rmse_std":       round(s["rmse"],     5),
            "r2_mean":        round(m["r2"],       5),
            "r2_std":         round(s["r2"],       5),
            "spearman_mean":  round(m["spearman"], 5),
            "spearman_std":   round(s["spearman"], 5),
            "n_folds":        len(result.fold_results),
            "n_landlords":    len(result.oof_true),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("mae_mean").reset_index(drop=True)
    return df


def suggest_best_model(
    comparison: pd.DataFrame,
    *,
    primary: str = "mae_mean",
    secondary: str = "spearman_mean",
) -> dict:
    """
    Recommend a model from a compare_models() table.

    Primary criterion: lowest MAE (easier to interpret scoring error).
    Tie-break: highest Spearman (ranking quality for recommendations).
    Also reports winners on RMSE / R² for transparency.

    Returns
    -------
    dict with keys: best_model, reason, winners_by_metric, comparison_sorted
    """
    if comparison.empty:
        return {
            "best_model": None,
            "reason": "No models to compare.",
            "winners_by_metric": {},
            "comparison_sorted": comparison,
        }

    lower_is_better = {"mae_mean", "rmse_mean"}
    higher_is_better = {"r2_mean", "spearman_mean"}

    winners: dict[str, str] = {}
    for metric in ("mae_mean", "rmse_mean", "r2_mean", "spearman_mean"):
        if metric not in comparison.columns:
            continue
        if metric in lower_is_better:
            winners[metric] = str(comparison.loc[comparison[metric].idxmin(), "model_type"])
        else:
            winners[metric] = str(comparison.loc[comparison[metric].idxmax(), "model_type"])

    ranked = comparison.copy()
    # Rank: primary ascending for MAE/RMSE, descending for R2/Spearman
    if primary in lower_is_better:
        ranked = ranked.sort_values(
            [primary, secondary],
            ascending=[True, False],
        ).reset_index(drop=True)
    else:
        ranked = ranked.sort_values(
            [primary, secondary],
            ascending=[False, False],
        ).reset_index(drop=True)

    best = ranked.iloc[0]
    best_name = str(best["model_type"])
    reason = (
        f"{best_name} is recommended: best {primary}={best[primary]:.5f} "
        f"(±{best.get(primary.replace('_mean', '_std'), float('nan')):.5f}), "
        f"{secondary}={best[secondary]:.5f}."
    )
    # Note near-ties
    if len(ranked) > 1:
        second = ranked.iloc[1]
        if primary in lower_is_better:
            gap = float(second[primary] - best[primary])
            gap_pct = gap / abs(float(best[primary])) * 100 if best[primary] else 0.0
        else:
            gap = float(best[primary] - second[primary])
            gap_pct = gap / abs(float(best[primary])) * 100 if best[primary] else 0.0
        if gap_pct < 2.0:
            reason += (
                f" Near-tie with {second['model_type']} "
                f"({primary} gap {gap_pct:.2f}%)."
            )

    return {
        "best_model": best_name,
        "reason": reason,
        "winners_by_metric": winners,
        "comparison_sorted": ranked,
        "best_row": best.to_dict(),
    }


def render_model_comparison_report(
    comparison: pd.DataFrame,
    suggestion: dict,
    *,
    title: str = "Landlord Model Comparison Report",
    n_landlords: int | None = None,
    n_folds: int | None = None,
    feature_set: str | None = None,
    extra_notes: list[str] | None = None,
) -> str:
    """Render a Markdown report comparing models and stating the recommendation."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated at: {pd.Timestamp.utcnow().isoformat()}_")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    best = suggestion.get("best_model")
    if best:
        lines.append(f"**Best model: `{best}`**")
        lines.append("")
        lines.append(suggestion.get("reason", ""))
    else:
        lines.append(suggestion.get("reason", "No recommendation available."))
    lines.append("")

    winners = suggestion.get("winners_by_metric") or {}
    if winners:
        lines.append("### Metric winners")
        lines.append("")
        lines.append("| Metric | Direction | Winner |")
        lines.append("| --- | --- | --- |")
        directions = {
            "mae_mean": "lower better",
            "rmse_mean": "lower better",
            "r2_mean": "higher better",
            "spearman_mean": "higher better",
        }
        labels = {
            "mae_mean": "MAE",
            "rmse_mean": "RMSE",
            "r2_mean": "R²",
            "spearman_mean": "Spearman",
        }
        for key, model in winners.items():
            lines.append(
                f"| {labels.get(key, key)} | {directions.get(key, '')} | `{model}` |"
            )
        lines.append("")

    lines.append("## Setup")
    lines.append("")
    if n_landlords is not None:
        lines.append(f"- Landlords scored: **{n_landlords}**")
    if n_folds is not None:
        lines.append(f"- CV: **GroupKFold**, {n_folds} folds (grouped by `LandLordID`)")
    if feature_set:
        lines.append(f"- Feature set: **{feature_set}**")
    lines.append("- Target: `AdjustedScore` (EB-shrunk mean company residual)")
    lines.append("- Models: CatBoost, XGBoost, LightGBM, Random Forest")
    lines.append("")

    lines.append("## Comparison table (sorted by MAE)")
    lines.append("")
    table = suggestion.get("comparison_sorted", comparison)
    if table is not None and not table.empty:
        display = table.copy()
        # Friendly column names for markdown
        rename = {
            "model_type": "Model",
            "mae_mean": "MAE",
            "mae_std": "MAE std",
            "rmse_mean": "RMSE",
            "rmse_std": "RMSE std",
            "r2_mean": "R²",
            "r2_std": "R² std",
            "spearman_mean": "Spearman",
            "spearman_std": "Spearman std",
            "n_folds": "Folds",
            "n_landlords": "N",
        }
        display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})
        lines.append(_df_to_md_table(display))
    else:
        lines.append("_No comparison rows._")
    lines.append("")

    lines.append("## How the best model was chosen")
    lines.append("")
    lines.append(
        "1. **Primary:** lowest mean Absolute Error (MAE) across GroupKFold folds — "
        "interpretable average scoring error."
    )
    lines.append(
        "2. **Tie-break:** highest Spearman rank correlation — preserves landlord "
        "ordering for recommendations."
    )
    lines.append(
        "3. RMSE and R² are reported for diagnostics; they do not override MAE "
        "unless MAE values are effectively tied (<2% relative gap)."
    )
    lines.append("")

    if extra_notes:
        lines.append("## Notes")
        lines.append("")
        for note in extra_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append(
        "_This report estimates predictive association with adjusted tenant outcomes; "
        "it does not prove causal landlord impact._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OOF residual summary
# ---------------------------------------------------------------------------

def oof_residual_summary(result: "CVResult") -> pd.DataFrame:
    """
    Per-landlord DataFrame with true score, OOF prediction, and residual.

    Parameters
    ----------
    result : CVResult from train_cv

    Returns
    -------
    pd.DataFrame columns: landlord_id, true_score, oof_pred, residual, abs_error
    Sorted descending by abs_error (largest errors first).
    """
    df = pd.DataFrame({
        "landlord_id": result.landlord_ids,
        "true_score":  result.oof_true,
        "oof_pred":    result.oof_predictions,
    })
    df["residual"]  = df["oof_pred"] - df["true_score"]
    df["abs_error"] = df["residual"].abs()
    return df.sort_values("abs_error", ascending=False).reset_index(drop=True)
