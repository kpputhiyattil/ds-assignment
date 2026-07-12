"""Packaged dual churn-decision artifact (with vs without interval features).

Bundles feature engineering + two model pairs:

* **with_interval**  — CORE + GAP + INTERVAL features
* **without_interval** — CORE + GAP only

So serving can compare Continue/Review/No-Go with and without the inferred
billing interval on the same customer history.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.config import Config, get_config
from src.features.build import (
    CORE_FEATURES,
    GAP_FEATURES,
    INTERVAL_FEATURES,
    build_scoring_features,
    feature_columns,
)
from src.features.interval import CONFIDENCE, INTERVAL
from src.serving.decision import decide
from src.serving.explanations import (
    compare_decisions,
    explain_inferred_interval,
    explain_model_drivers,
    explain_recommendation,
)


class _ModelPair:
    def __init__(self, frequency, severity, raw_frequency, feature_cols: list[str]):
        self.frequency = frequency
        self.severity = severity
        self.raw_frequency = raw_frequency
        self.feature_cols = feature_cols
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(
                self.raw_frequency, feature_perturbation="tree_path_dependent"
            )
        return self._explainer

    def score(self, feat_df: pl.DataFrame) -> dict:
        X = feat_df.select(self.feature_cols).to_numpy().astype(float)
        p = self.frequency.predict_proba(X)[:, 1]
        sev = np.clip(self.severity.predict(X), 0, None)
        return {"X": X, "churn_prob": p, "expected_dollar_churn": p * sev}


class ChurnDecisionModel:
    """Dual-model scoring + explanation + decision bundle."""

    def __init__(
        self,
        with_interval: _ModelPair,
        without_interval: _ModelPair,
        cfg: Config,
        version: str,
        git_commit: str,
        comparison_metrics: dict | None = None,
    ):
        self.with_interval = with_interval
        self.without_interval = without_interval
        # Back-compat aliases used by older checklist / health code.
        self.frequency = with_interval.frequency
        self.severity = with_interval.severity
        self.raw_frequency = with_interval.raw_frequency
        self.feature_cols = with_interval.feature_cols
        self.cfg = cfg
        self.version = version
        self.git_commit = git_commit
        self.comparison_metrics = comparison_metrics or {}
        self._explainer = None

    def has_feature_engineering(self) -> bool:
        return build_scoring_features is not None

    def has_models(self) -> bool:
        return (
            self.with_interval.frequency is not None
            and self.without_interval.frequency is not None
        )

    def has_explainer(self) -> bool:
        return self.with_interval.raw_frequency is not None

    @property
    def explainer(self):
        return self.with_interval.explainer

    def _row_payload(
        self,
        *,
        cid: str,
        snapshot: str,
        feat_row: dict,
        scored_with: dict,
        scored_wo: dict,
        idx: int,
        briefing_with: dict | None,
        briefing_wo: dict | None,
    ) -> dict:
        interval = feat_row[INTERVAL]
        conf = float(feat_row[CONFIDENCE])
        p_w = float(scored_with["churn_prob"][idx])
        p_o = float(scored_wo["churn_prob"][idx])
        exp_w = float(scored_with["expected_dollar_churn"][idx])
        exp_o = float(scored_wo["expected_dollar_churn"][idx])

        # Decision thresholds come from live config (not the pickled package cfg)
        # so Continue/No-Go bands can be tuned without re-packaging models.
        decision_cfg = get_config()
        dec_w = decide(p_w, conf, decision_cfg)
        # Without-interval model still uses the same confidence guardrail for
        # fair Continue/No-Go automation (data-quality is independent of FE set).
        dec_o = decide(p_o, conf, decision_cfg)

        diag = {
            "inferred_interval": interval,
            "interval_confidence": round(conf, 4),
            "n_events": feat_row.get("n_events"),
            "n_gaps": feat_row.get("n_gaps"),
            "median_gap_days": feat_row.get("median_gap_days"),
            "gap_cv": feat_row.get("gap_cv"),
            "dominant_base": feat_row.get("dominant_base"),
            "dominant_share": feat_row.get("dominant_share"),
            "n_distinct_bases": feat_row.get("n_distinct_bases"),
            "recency_days": feat_row.get("recency_days"),
            "tenure_days": feat_row.get("tenure_days"),
        }
        interval_reason = explain_inferred_interval(diag)

        with_block = {
            "recommendation": dec_w["recommendation"],
            "reason": dec_w["reason"],
            "detail": dec_w["detail"],
            "auto_decided": dec_w["auto_decided"],
            "churn_probability": round(p_w, 4),
            "expected_dollar_churn": round(exp_w, 2),
            "decision_reason": explain_recommendation(
                recommendation=dec_w["recommendation"],
                reason=dec_w["reason"],
                detail=dec_w["detail"],
                churn_probability=p_w,
                interval_confidence=conf,
                inferred_interval=interval,
                model_tag="trained WITH inferred-interval features",
            ),
            "model_drivers": explain_model_drivers(briefing_with),
            "briefing": briefing_with,
        }
        without_block = {
            "recommendation": dec_o["recommendation"],
            "reason": dec_o["reason"],
            "detail": dec_o["detail"],
            "auto_decided": dec_o["auto_decided"],
            "churn_probability": round(p_o, 4),
            "expected_dollar_churn": round(exp_o, 2),
            "decision_reason": explain_recommendation(
                recommendation=dec_o["recommendation"],
                reason=dec_o["reason"],
                detail=dec_o["detail"],
                churn_probability=p_o,
                interval_confidence=conf,
                inferred_interval=interval,
                model_tag="trained WITHOUT inferred-interval features",
            ),
            "model_drivers": explain_model_drivers(briefing_wo),
            "briefing": briefing_wo,
        }

        # Top-level fields keep "with interval" as the primary serving decision.
        # Flat aliases are included so UIs/CSV always see both model outputs.
        return {
            "customer_id": cid,
            "snapshot": snapshot,
            "inferred_interval": interval,
            "interval_confidence": round(conf, 4),
            "interval_reason": interval_reason,
            "recommendation": with_block["recommendation"],
            "reason": with_block["reason"],
            "detail": with_block["detail"],
            "auto_decided": with_block["auto_decided"],
            "churn_probability": with_block["churn_probability"],
            "expected_dollar_churn": with_block["expected_dollar_churn"],
            "decision_reason": with_block["decision_reason"],
            "with_interval": with_block,
            "without_interval": without_block,
            "with_recommendation": with_block["recommendation"],
            "with_p_churn": with_block["churn_probability"],
            "with_decision_reason": with_block["decision_reason"],
            "without_recommendation": without_block["recommendation"],
            "without_p_churn": without_block["churn_probability"],
            "without_decision_reason": without_block["decision_reason"],
            "recommendations_agree": with_block["recommendation"]
            == without_block["recommendation"],
            "comparison_narrative": compare_decisions(
                with_block["recommendation"],
                without_block["recommendation"],
                p_w,
                p_o,
            ),
            "comparison": {
                "recommendations_agree": with_block["recommendation"]
                == without_block["recommendation"],
                "churn_probability_delta": round(p_w - p_o, 4),
                "expected_dollar_churn_delta": round(exp_w - exp_o, 2),
                "narrative": compare_decisions(
                    with_block["recommendation"],
                    without_block["recommendation"],
                    p_w,
                    p_o,
                ),
            },
            "model_version": self.version,
            "n_events": diag["n_events"],
            "median_gap_days": diag["median_gap_days"],
        }

    def assess(
        self, events: pl.DataFrame, snapshot: dt.date, customer_id: str | None = None,
        explain: bool = True,
    ) -> dict:
        out = self.assess_many(
            events,
            snapshot,
            explain_top_n=1 if explain else 0,
            explain_ids=[customer_id] if customer_id and explain else None,
            customer_ids=[customer_id] if customer_id else None,
        )
        if not out["results"]:
            return {
                "customer_id": customer_id or "",
                "snapshot": str(snapshot),
                "recommendation": "Review",
                "reason": "no_history",
                "detail": "No pre-snapshot invoice history to score.",
                "auto_decided": False,
                "churn_probability": None,
                "expected_dollar_churn": None,
                "inferred_interval": None,
                "interval_confidence": None,
                "interval_reason": (
                    "No pre-snapshot invoices were available, so no billing "
                    "interval could be inferred."
                ),
                "decision_reason": explain_recommendation(
                    recommendation="Review",
                    reason="no_history",
                    detail="",
                    churn_probability=None,
                    interval_confidence=None,
                ),
                "model_version": self.version,
                "briefing": None,
            }
        row = out["results"][0]
        # Preserve briefing key for older API clients.
        row["briefing"] = (row.get("with_interval") or {}).get("briefing")
        return row

    def assess_many(
        self,
        events: pl.DataFrame,
        snapshot: dt.date,
        *,
        explain_top_n: int = 0,
        explain_ids: list[str] | None = None,
        customer_ids: list[str] | None = None,
    ) -> dict:
        from src.explainability.shap_utils import explain_row

        cust_col = self.cfg.schema_.customer_id_col
        feats = build_scoring_features(events, snapshot, self.cfg)
        if customer_ids is not None:
            feats = feats.filter(pl.col(cust_col).is_in(customer_ids))
        if feats.height == 0:
            return {
                "snapshot": str(snapshot),
                "n_customers": 0,
                "results": [],
                "summary": {},
                "explanations": [],
                "comparison_summary": {},
                "model_version": self.version,
            }

        scored_w = self.with_interval.score(feats)
        scored_o = self.without_interval.score(feats)
        ids = feats[cust_col].to_list()
        feat_rows = feats.to_dicts()

        explain_set: set[str] = set(explain_ids or [])
        if explain_top_n > 0:
            order = np.argsort(-scored_w["churn_prob"])[:explain_top_n]
            explain_set.update(ids[j] for j in order)

        briefings_w: dict[str, dict] = {}
        briefings_o: dict[str, dict] = {}
        id_to_idx = {cid: i for i, cid in enumerate(ids)}
        for cid in explain_set:
            i = id_to_idx.get(cid)
            if i is None:
                continue
            briefings_w[cid] = explain_row(
                self.with_interval.explainer,
                scored_w["X"][i],
                self.with_interval.feature_cols,
                float(scored_w["churn_prob"][i]),
                self.cfg,
            )
            briefings_o[cid] = explain_row(
                self.without_interval.explainer,
                scored_o["X"][i],
                self.without_interval.feature_cols,
                float(scored_o["churn_prob"][i]),
                self.cfg,
            )

        results = []
        for i, cid in enumerate(ids):
            results.append(
                self._row_payload(
                    cid=cid,
                    snapshot=str(snapshot),
                    feat_row=feat_rows[i],
                    scored_with=scored_w,
                    scored_wo=scored_o,
                    idx=i,
                    briefing_with=briefings_w.get(cid),
                    briefing_wo=briefings_o.get(cid),
                )
            )

        reco_w: dict[str, int] = {}
        reco_o: dict[str, int] = {}
        agree = 0
        for r in results:
            rw = r["with_interval"]["recommendation"]
            ro = r["without_interval"]["recommendation"]
            reco_w[rw] = reco_w.get(rw, 0) + 1
            reco_o[ro] = reco_o.get(ro, 0) + 1
            if rw == ro:
                agree += 1

        n = len(results)
        comparison_summary = {
            "n_customers": n,
            "agreement_rate": round(agree / max(n, 1), 4),
            "n_disagree": n - agree,
            "recommendation_counts_with_interval": reco_w,
            "recommendation_counts_without_interval": reco_o,
            "mean_churn_probability_with": round(
                float(np.mean(scored_w["churn_prob"])), 4
            ),
            "mean_churn_probability_without": round(
                float(np.mean(scored_o["churn_prob"])), 4
            ),
            "mean_churn_probability_delta": round(
                float(np.mean(scored_w["churn_prob"] - scored_o["churn_prob"])), 4
            ),
            "offline_holdout_metrics": self.comparison_metrics,
            "narrative": (
                f"On this file, the with-interval and without-interval models "
                f"agree on {agree}/{n} customers ({100 * agree / max(n, 1):.1f}%). "
                f"Mean P(churn) is {float(np.mean(scored_w['churn_prob'])):.1%} with "
                f"interval vs {float(np.mean(scored_o['churn_prob'])):.1%} without "
                f"(Δ={float(np.mean(scored_w['churn_prob'] - scored_o['churn_prob'])):+.1%})."
            ),
        }

        summary = {
            "n_customers": n,
            "recommendation_counts": reco_w,
            "recommendation_counts_without_interval": reco_o,
            "mean_churn_probability": comparison_summary["mean_churn_probability_with"],
            "total_expected_dollar_churn": round(
                float(np.sum(scored_w["expected_dollar_churn"])), 2
            ),
            "mean_interval_confidence": round(
                float(np.mean([r["interval_confidence"] for r in results])), 4
            ),
            "n_explained": len(explain_set),
            "agreement_rate": comparison_summary["agreement_rate"],
        }

        # Explanations list = rows that received SHAP briefings (still full dual payload).
        explanations = [r for r in results if r["customer_id"] in explain_set]

        return {
            "snapshot": str(snapshot),
            "n_customers": n,
            "results": results,
            "summary": summary,
            "comparison_summary": comparison_summary,
            "explanations": explanations,
            "model_version": self.version,
        }
