"""Ablation harness -- does the inferred interval add value? (Step 6).

Two ablations, each a paired base-vs-treatment comparison on the *same*
temporal test set (train on the earlier snapshot, test on the later one):

* **Ablation 1 (core value)**      base = CORE          vs  CORE + INTERVAL
* **Ablation 2 (beyond raw gaps)** base = CORE + GAP    vs  CORE + GAP + INTERVAL

Ablation 2 is the stricter test: it asks whether the *derived* interval adds
anything over the raw gap statistics it is computed from.

For speed the ablation models are lean, uncalibrated LightGBMs (calibration is
orthogonal and applied equally to base and treatment; ranking metrics are
unaffected). Predictions are cached so the significance analysis can run as a
separate, fast step.

"Good enough" is pre-registered (decided before seeing results): the interval
ships iff it delivers a **statistically significant** improvement in PR-AUC
*or* dollar-recall@10% with **no material degradation** in Brier or expected-
loss MAE -- evaluated on both the full cohort and the recurring (>=2 event)
subsegment.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import Config, get_config
from src.features.build import (
    CORE_FEATURES,
    GAP_FEATURES,
    INTERVAL_FEATURES,
)
from src.features.labels import Y_CHURN, Y_LOSS
from src.models import metrics as M
from src.models.stats import delong_roc_test, paired_bootstrap_delta
from src.models.train import temporal_split

logger = logging.getLogger(__name__)

# Lean models for a fast, fair base-vs-treatment comparison.
_FREQ = dict(objective="binary", n_estimators=200, learning_rate=0.08,
             num_leaves=63, subsample=0.8, colsample_bytree=0.8)
_SEV = dict(objective="regression_l1", n_estimators=200, learning_rate=0.08,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8)

FEATURE_SETS = {
    "core": CORE_FEATURES,
    "core_interval": CORE_FEATURES + INTERVAL_FEATURES,
    "core_gap": CORE_FEATURES + GAP_FEATURES,
    "core_gap_interval": CORE_FEATURES + GAP_FEATURES + INTERVAL_FEATURES,
}

ABLATIONS = {
    "A1_core_value": ("core", "core_interval"),
    "A2_beyond_raw_gaps": ("core_gap", "core_gap_interval"),
}

PREDS_PATH = "ablation_preds.parquet"
REPORT_PATH = "ablation_report.json"


def _train_predict(feats, train_df, test_df, cfg):
    """Train lean freq+sev on a feature set; return (p_churn, expected_churn) on test."""
    Xtr = train_df.select(feats).to_numpy().astype(float)
    ytr = train_df[Y_CHURN].to_numpy().astype(int)
    losstr = train_df[Y_LOSS].to_numpy().astype(float)
    Xte = test_df.select(feats).to_numpy().astype(float)

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    freq = LGBMClassifier(**_FREQ, scale_pos_weight=spw,
                          random_state=cfg.random_seed, n_jobs=2, verbose=-1)
    freq.fit(Xtr, ytr)
    sev = LGBMRegressor(**_SEV, random_state=cfg.random_seed, n_jobs=2, verbose=-1)
    sev.fit(Xtr[ytr == 1], losstr[ytr == 1])

    p = freq.predict_proba(Xte)[:, 1]
    exp = p * np.clip(sev.predict(Xte), 0, None)
    return p, exp


def compute_predictions(cfg: Config | None = None) -> str:
    """Train every feature set and cache test predictions for analysis."""
    cfg = cfg or get_config()
    df = pl.read_parquet(cfg.paths.resolve(cfg.paths.processed_dir) / "features.parquet")
    train_df, test_df = temporal_split(df, cfg)

    out = {
        Y_CHURN: test_df[Y_CHURN].to_numpy().astype(int),
        Y_LOSS: test_df[Y_LOSS].to_numpy().astype(float),
        "recurring": (test_df["n_events"].to_numpy() >= 2).astype(int),
    }
    for name, feats in FEATURE_SETS.items():
        p, exp = _train_predict(feats, train_df, test_df, cfg)
        out[f"p_{name}"] = p
        out[f"exp_{name}"] = exp
        logger.info("Trained feature set '%s' (%d features)", name, len(feats))

    preds = pl.DataFrame(out)
    path = cfg.paths.resolve(cfg.paths.processed_dir) / PREDS_PATH
    preds.write_parquet(path)
    logger.info("Cached ablation predictions -> %s", path)
    return str(path)


def _analyze_segment(y, loss, p_base, p_treat, exp_base, exp_treat, seed, n_boot=150) -> dict:
    """All significance tests for one ablation on one segment."""
    n = len(y)

    def pr(idx):
        return M.pr_auc(y[idx], p_base[idx]), M.pr_auc(y[idx], p_treat[idx])

    def rec(idx):
        return (M.dollar_recall_at_k(loss[idx], exp_base[idx], 0.10),
                M.dollar_recall_at_k(loss[idx], exp_treat[idx], 0.10))

    def brier(idx):
        return M.brier(y[idx], p_base[idx]), M.brier(y[idx], p_treat[idx])

    def emae(idx):
        return M.mae(loss[idx], exp_base[idx]), M.mae(loss[idx], exp_treat[idx])

    return {
        "n": int(n),
        "roc_auc_delong": delong_roc_test(y, p_base, p_treat),
        "pr_auc": paired_bootstrap_delta(pr, n, n_boot=n_boot, seed=seed),
        "dollar_recall_top10pct": paired_bootstrap_delta(rec, n, n_boot=n_boot, seed=seed),
        "brier": paired_bootstrap_delta(brier, n, n_boot=n_boot, seed=seed),
        "expected_loss_mae": paired_bootstrap_delta(emae, n, n_boot=n_boot, seed=seed),
    }


def _verdict(seg: dict) -> dict:
    """Pre-registered 'good enough' rule applied to a segment's results."""
    pr_better = seg["pr_auc"]["delta"] > 0 and seg["pr_auc"]["significant"]
    rec_better = (
        seg["dollar_recall_top10pct"]["delta"] > 0
        and seg["dollar_recall_top10pct"]["significant"]
    )
    # 'No material degradation': Brier not worse by >0.005, MAE not worse by >1%.
    brier_ok = seg["brier"]["delta"] <= 0.005
    mae_ok = seg["expected_loss_mae"]["delta"] <= 0.01 * abs(seg["expected_loss_mae"]["base"])
    good = bool((pr_better or rec_better) and brier_ok and mae_ok)
    return {
        "improves_pr_auc": pr_better,
        "improves_dollar_recall": rec_better,
        "no_material_brier_degradation": bool(brier_ok),
        "no_material_mae_degradation": bool(mae_ok),
        "good_enough": good,
    }


def analyze(cfg: Config | None = None, n_boot: int = 150) -> dict:
    """Run both ablations on overall + recurring segments; write the report."""
    cfg = cfg or get_config()
    preds = pl.read_parquet(cfg.paths.resolve(cfg.paths.processed_dir) / PREDS_PATH)
    y = preds[Y_CHURN].to_numpy()
    loss = preds[Y_LOSS].to_numpy()
    rec_mask = preds["recurring"].to_numpy().astype(bool)

    report = {"ablations": {}}
    for ab_name, (base, treat) in ABLATIONS.items():
        pb, pt = preds[f"p_{base}"].to_numpy(), preds[f"p_{treat}"].to_numpy()
        eb, et = preds[f"exp_{base}"].to_numpy(), preds[f"exp_{treat}"].to_numpy()

        overall = _analyze_segment(y, loss, pb, pt, eb, et, cfg.random_seed, n_boot)
        overall["verdict"] = _verdict(overall)

        ry, rl = y[rec_mask], loss[rec_mask]
        recurring = _analyze_segment(
            ry, rl, pb[rec_mask], pt[rec_mask], eb[rec_mask], et[rec_mask],
            cfg.random_seed, n_boot,
        )
        recurring["verdict"] = _verdict(recurring)

        report["ablations"][ab_name] = {
            "base": base, "treatment": treat,
            "overall": overall, "recurring": recurring,
        }
        logger.info(
            "%s | overall good_enough=%s | recurring good_enough=%s",
            ab_name, overall["verdict"]["good_enough"],
            recurring["verdict"]["good_enough"],
        )

    # Headline: interval is 'good enough' if EITHER ablation passes on EITHER
    # segment with significant improvement and no degradation.
    passes = [
        seg["verdict"]["good_enough"]
        for ab in report["ablations"].values()
        for seg in (ab["overall"], ab["recurring"])
    ]
    report["interval_good_enough"] = bool(any(passes))
    report["decision"] = (
        "SHIP the inferred interval as a substitute for the missing real interval"
        if report["interval_good_enough"]
        else "DO NOT ship: invoice-derived features already capture the signal"
    )

    path = cfg.paths.resolve(cfg.paths.artifacts_dir) / REPORT_PATH
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Verdict: %s -> %s", report["interval_good_enough"], report["decision"])
    return report


def run(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    compute_predictions(cfg)
    return analyze(cfg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()


if __name__ == "__main__":
    main()
