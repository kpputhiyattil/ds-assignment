"""Packaged churn-decision artifact (Step 8).

A single deployable object that bundles the four pieces that must never drift
apart: **feature engineering**, the **models** (calibrated frequency + severity),
the **SHAP explainer**, and the **decision logic**. Given a customer's raw
invoices and a scoring date it returns the full assessment: churn probability,
expected dollar churn, inferred interval, recommendation, and a briefing.

The SHAP explainer is reconstructed deterministically from the packaged raw
model on first use (TreeExplainer is cheap to build and fragile to pickle), so
the explainability capability travels with the artifact without a brittle
serialized explainer.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.config import Config, get_config
from src.features.build import build_scoring_features, feature_columns
from src.features.interval import CONFIDENCE, INTERVAL
from src.serving.decision import decide


class ChurnDecisionModel:
    """Self-contained scoring + explanation + decision bundle."""

    def __init__(self, frequency, severity, raw_frequency, feature_cols,
                 cfg: Config, version: str, git_commit: str):
        self.frequency = frequency          # calibrated P(churn)
        self.severity = severity            # E(loss | churn)
        self.raw_frequency = raw_frequency  # for SHAP
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.version = version
        self.git_commit = git_commit
        self._explainer = None              # lazily built

    # --- capabilities (skill packaging checklist) ---
    def has_feature_engineering(self) -> bool:
        return build_scoring_features is not None

    def has_models(self) -> bool:
        return self.frequency is not None and self.severity is not None

    def has_explainer(self) -> bool:
        return self.raw_frequency is not None

    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(
                self.raw_frequency, feature_perturbation="tree_path_dependent"
            )
        return self._explainer

    def _score_matrix(self, feat_df: pl.DataFrame) -> dict:
        X = feat_df.select(self.feature_cols).to_numpy().astype(float)
        p = self.frequency.predict_proba(X)[:, 1]
        sev = np.clip(self.severity.predict(X), 0, None)
        return {"X": X, "churn_prob": p, "expected_dollar_churn": p * sev,
                "severity": sev}

    def assess(
        self, events: pl.DataFrame, snapshot: dt.date, customer_id: str | None = None,
        explain: bool = True,
    ) -> dict:
        """Full assessment for one customer's raw invoices as of ``snapshot``."""
        cust_col = self.cfg.schema_.customer_id_col
        feats = build_scoring_features(events, snapshot, self.cfg)
        if customer_id is not None:
            feats = feats.filter(pl.col(cust_col) == customer_id)
        if feats.height == 0:
            return {"recommendation": "Review", "reason": "no_history",
                    "detail": "No pre-snapshot invoice history to score."}

        row = feats.row(0, named=True)
        scored = self._score_matrix(feats.head(1))
        prob = float(scored["churn_prob"][0])
        interval = row[INTERVAL]
        conf = float(row[CONFIDENCE])

        decision = decide(prob, conf, self.cfg)
        out = {
            "customer_id": row[cust_col],
            "snapshot": str(snapshot),
            "churn_probability": round(prob, 4),
            "expected_dollar_churn": round(float(scored["expected_dollar_churn"][0]), 2),
            "inferred_interval": interval,
            "interval_confidence": round(conf, 4),
            **decision,
            "model_version": self.version,
        }
        if explain:
            from src.explainability.shap_utils import explain_row
            out["briefing"] = explain_row(
                self.explainer, scored["X"][0], self.feature_cols, prob, self.cfg
            )
        return out
