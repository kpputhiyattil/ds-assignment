"""Two-part dollar-churn model training (Step 5).

Expected dollar churn = P(churn) x E(loss | churn):

* **Frequency** -- LightGBM classifier for ``P(churn)``, class-weighted for the
  ~89% base rate and **probability-calibrated** (isotonic) so the output is a
  usable probability for the credit decision.
* **Severity** -- LightGBM L1 regressor for ``E(loss | churn)``, trained on
  churned customers only (robust to the heavily right-skewed dollar target).

Validation is **temporal**: train on earlier snapshot(s), test on the latest.
A class-weighted logistic-regression pipeline is trained as an interpretable
baseline. Everything (models, feature list, hyperparameters, metrics, data
snapshot, git commit) is written to a versioned artifact directory.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import Config, get_config
from src.features.build import feature_columns
from src.features.labels import Y_CHURN, Y_LOSS
from src.models import metrics as M

logger = logging.getLogger(__name__)

DOLLAR = "dollar_at_risk"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def temporal_split(df: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on all but the latest snapshot; test on the latest."""
    snaps = sorted(cfg.snapshots)
    test_snap = snaps[-1]
    train = df.filter(pl.col("snapshot_date") != pl.lit(test_snap).cast(pl.Date))
    test = df.filter(pl.col("snapshot_date") == pl.lit(test_snap).cast(pl.Date))
    return train, test


def _xy(df: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df.select(feats).to_numpy().astype(float)
    y = df[Y_CHURN].to_numpy().astype(int)
    loss = df[Y_LOSS].to_numpy().astype(float)
    return X, y, loss


def train_frequency(X, y, cfg: Config):
    """Fit a class-weighted LightGBM classifier, then isotonic-calibrate it."""
    params = dict(cfg.model.frequency)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)  # balances the majority-positive skew

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_seed, stratify=y
    )
    raw = LGBMClassifier(
        **params, scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_seed, n_jobs=2, verbose=-1,
    )
    raw.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic")
    calibrated.fit(X_cal, y_cal)
    return calibrated, raw, scale_pos_weight


def train_severity(X, y, loss, cfg: Config) -> LGBMRegressor:
    """Fit an L1 LightGBM regressor for E(loss | churn) on churned rows only."""
    mask = y == 1
    params = dict(cfg.model.severity)
    reg = LGBMRegressor(
        **params, random_state=cfg.random_seed, n_jobs=2, verbose=-1
    )
    reg.fit(X[mask], loss[mask])
    return reg


def train_baseline(X, y, cfg: Config) -> Pipeline:
    """Interpretable class-weighted logistic-regression baseline."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    return pipe


def _freq_metrics(y, p) -> dict:
    return {
        "roc_auc": round(M.roc_auc(y, p), 4),
        "pr_auc": round(M.pr_auc(y, p), 4),
        "brier": round(M.brier(y, p), 4),
    }


def evaluate(freq, raw, sev, baseline, X, y, loss, mean_train_loss: float) -> dict:
    """Compute frequency, severity, and combined dollar-churn metrics."""
    p_cal = freq.predict_proba(X)[:, 1]
    p_raw = raw.predict_proba(X)[:, 1]
    p_base = baseline.predict_proba(X)[:, 1]

    churn_mask = y == 1
    sev_pred_all = np.clip(sev.predict(X), 0, None)
    expected_churn = p_cal * sev_pred_all

    out = {
        "frequency": {
            "calibrated": _freq_metrics(y, p_cal),
            "raw": _freq_metrics(y, p_raw),
            "baseline_logistic": _freq_metrics(y, p_base),
            "base_rate": round(float(y.mean()), 4),
        },
        "severity": {
            "mae_on_churned": round(M.mae(loss[churn_mask], sev_pred_all[churn_mask]), 2),
            "baseline_mae_on_churned": round(
                M.mae(loss[churn_mask], np.full(int(churn_mask.sum()), mean_train_loss)), 2
            ),
        },
        "dollar_churn": {
            "expected_vs_actual_mae": round(M.mae(loss, expected_churn), 2),
            "dollar_recall_top5pct": round(M.dollar_recall_at_k(loss, expected_churn, 0.05), 4),
            "dollar_recall_top10pct": round(M.dollar_recall_at_k(loss, expected_churn, 0.10), 4),
            "pct_dollars_captured_top10pct": round(
                M.pct_dollars_captured(loss, expected_churn, 0.10), 2
            ),
        },
        "n": int(len(y)),
    }
    return out


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    feats = feature_columns()
    df = pl.read_parquet(cfg.paths.resolve(cfg.paths.processed_dir) / "features.parquet")
    train_df, test_df = temporal_split(df, cfg)

    X_tr, y_tr, loss_tr = _xy(train_df, feats)
    X_te, y_te, loss_te = _xy(test_df, feats)
    mean_train_loss = float(loss_tr[y_tr == 1].mean())

    logger.info("Train n=%d (churn=%.3f) | Test n=%d (churn=%.3f)",
                len(y_tr), y_tr.mean(), len(y_te), y_te.mean())

    freq, raw, spw = train_frequency(X_tr, y_tr, cfg)
    sev = train_severity(X_tr, y_tr, loss_tr, cfg)
    baseline = train_baseline(X_tr, y_tr, cfg)

    overall = evaluate(freq, raw, sev, baseline, X_te, y_te, loss_te, mean_train_loss)

    # Subsegment readout: recurring customers (>=2 events).
    sub = test_df.filter(pl.col("n_events") >= 2)
    Xs, ys, losss = _xy(sub, feats)
    subseg = evaluate(freq, raw, sev, baseline, Xs, ys, losss, mean_train_loss)

    # --- Versioned artifact ---
    version = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "models" / f"model_{version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(freq, model_dir / "frequency.joblib")
    joblib.dump(raw, model_dir / "frequency_raw.joblib")
    joblib.dump(sev, model_dir / "severity.joblib")
    joblib.dump(baseline, model_dir / "baseline.joblib")

    metadata = {
        "version": version,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "train_snapshots": [str(s) for s in sorted(cfg.snapshots)[:-1]],
        "test_snapshot": str(sorted(cfg.snapshots)[-1]),
        "feature_columns": feats,
        "scale_pos_weight": round(spw, 4),
        "hyperparameters": {"frequency": dict(cfg.model.frequency),
                            "severity": dict(cfg.model.severity)},
        "cohort": {"train": len(y_tr), "test": len(y_te), "test_recurring": len(ys)},
        "metrics_overall": overall,
        "metrics_recurring_subsegment": subseg,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (cfg.paths.resolve(cfg.paths.artifacts_dir) / "models" / "latest.txt").write_text(
        version, encoding="utf-8"
    )
    logger.info("Saved model %s | test ROC-AUC(cal)=%.3f PR-AUC=%.3f $recall@10%%=%.3f",
                version, overall["frequency"]["calibrated"]["roc_auc"],
                overall["frequency"]["calibrated"]["pr_auc"],
                overall["dollar_churn"]["dollar_recall_top10pct"])
    return metadata


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()


if __name__ == "__main__":
    main()
