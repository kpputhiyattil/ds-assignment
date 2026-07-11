"""TreeSHAP explainability for the frequency (churn) model (Step 7).

Credit decisions affect a customer, so per-prediction explainability is
non-optional. We use **TreeSHAP** (exact, fast for LightGBM) on the *raw*
frequency model -- the calibrated wrapper is a monotone transform on top, so the
raw margin drives the ranking and is what SHAP explains.

Two outputs:

* **Global**: mean(|SHAP|) per feature + a beeswarm summary plot.
* **Per-customer briefing**: the plain-language format a credit analyst reads --
  headline (risk + tier + confidence), main reason, factors increasing vs
  decreasing churn risk (each with its actual value), and an interpretation. The
  raw SHAP vector is kept underneath for anyone who wants to dig in.

Perturbation is ``tree_path_dependent`` (fast, no background set); switch to
``interventional`` with a background sample if strict marginal attributions are
needed. SHAP explains the *model*, not real-world causation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import shap

from src.config import Config, get_config
from src.features.build import feature_columns

logger = logging.getLogger(__name__)

# Feature column -> plain-language phrase for the briefing.
HUMAN_NAMES = {
    "recency_days": "days since last invoice",
    "tenure_days": "customer tenure",
    "n_events": "number of billing events",
    "n_events_past12": "billing events in the last 12 months",
    "total_amount": "lifetime revenue",
    "past12_amount": "revenue in the last 12 months",
    "prior12_amount": "revenue in the prior 12 months",
    "mean_amount": "average invoice amount",
    "last_amount": "most recent invoice amount",
    "max_amount": "largest invoice amount",
    "n_invoices_total": "total invoice lines",
    "revenue_trend": "revenue trend (recent vs prior year)",
    "amount_cv": "invoice-amount variability",
    "median_gap_days": "typical days between invoices",
    "gap_cv": "billing regularity",
    "n_gaps": "observed billing cycles",
    "interval_confidence": "billing-interval confidence",
    "dominant_share": "dominant-cadence share",
    "n_distinct_bases": "competing cadences",
    "is_insufficient_history": "flag: insufficient history",
    "is_one_time_candidate": "flag: likely one-time",
    "is_monthly": "flag: monthly cadence",
    "is_quarterly": "flag: quarterly cadence",
    "is_semi_annual": "flag: semi-annual cadence",
    "is_annual": "flag: annual cadence",
    "is_mixed": "flag: mixed cadence",
    "is_irregular": "flag: irregular cadence",
}


def load_latest_model(cfg: Config) -> tuple[object, dict]:
    """Load the raw frequency model + metadata for the latest trained version."""
    models_dir = cfg.paths.resolve(cfg.paths.artifacts_dir) / "models"
    version = (models_dir / "latest.txt").read_text(encoding="utf-8").strip()
    mdir = models_dir / f"model_{version}"
    model = joblib.load(mdir / "frequency_raw.joblib")
    meta = json.loads((mdir / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


def _normalize_shap(sv) -> np.ndarray:
    """Return a 2D (n, n_features) SHAP array for the positive class."""
    if isinstance(sv, list):  # [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:          # (n, n_features, n_classes)
        sv = sv[:, :, -1]
    return sv


def _fmt_value(feat: str, val: float) -> str:
    if feat.startswith("is_"):
        return "yes" if val >= 0.5 else "no"
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    if feat in {"recency_days", "tenure_days", "median_gap_days", "n_events",
                "n_events_past12", "n_gaps", "n_invoices_total", "n_distinct_bases"}:
        return f"{val:.0f}"
    return f"{val:,.2f}"


def _tier(prob: float, cfg: Config) -> str:
    if prob >= cfg.decision.nogo_min_risk:
        return "High risk"
    if prob <= cfg.decision.continue_max_risk:
        return "Low risk"
    return "Medium risk"


def _confidence(prob: float) -> str:
    """Confidence = distance of P(churn) from the 0.5 decision boundary."""
    d = abs(prob - 0.5) * 2  # 0 at boundary, 1 at extremes
    label = "High" if d >= 0.6 else "Medium" if d >= 0.3 else "Low"
    return f"{label} (|p-0.5| margin {d:.2f})"


def explain_row(
    explainer, x_row: np.ndarray, feats: list[str], prob: float, cfg: Config,
    top_k: int = 4,
) -> dict:
    """Per-customer SHAP briefing as a structured dict."""
    sv = _normalize_shap(explainer.shap_values(x_row.reshape(1, -1)))[0]
    order = np.argsort(-np.abs(sv))
    increasing, decreasing = [], []
    for i in order:
        item = {
            "feature": feats[i],
            "label": HUMAN_NAMES.get(feats[i], feats[i]),
            "value": _fmt_value(feats[i], float(x_row[i])),
            "shap": round(float(sv[i]), 4),
        }
        if sv[i] > 0 and len(increasing) < top_k:
            increasing.append(item)
        elif sv[i] < 0 and len(decreasing) < top_k:
            decreasing.append(item)
    main = {"feature": feats[order[0]], "label": HUMAN_NAMES.get(feats[order[0]], feats[order[0]]),
            "direction": "increasing" if sv[order[0]] > 0 else "decreasing"}
    return {
        "churn_probability": round(float(prob), 4),
        "tier": _tier(prob, cfg),
        "confidence": _confidence(prob),
        "main_reason": main,
        "risk_increasing": increasing,
        "risk_decreasing": decreasing,
    }


def format_briefing(customer_id: str, b: dict) -> str:
    """Render the structured briefing as the plain-language analyst format."""
    lines = [
        f"{customer_id} - P(churn): {b['churn_probability']:.2f} ({b['tier']})",
        f"Confidence: {b['confidence']}",
        "",
        f"Main reason: {b['main_reason']['label']} is the single biggest factor "
        f"{b['main_reason']['direction']} churn risk.",
        "",
        "Factors increasing churn risk:",
    ]
    lines += [f"  - {it['label']} ({it['value']})  [shap {it['shap']:+.3f}]"
              for it in b["risk_increasing"]] or ["  - (none)"]
    lines += ["", "Factors decreasing churn risk:"]
    lines += [f"  - {it['label']} ({it['value']})  [shap {it['shap']:+.3f}]"
              for it in b["risk_decreasing"]] or ["  - (none)"]
    return "\n".join(lines)


def run(cfg: Config | None = None, sample_n: int = 5000) -> dict:
    """Compute global importance + a summary plot + example briefings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or get_config()
    feats = feature_columns()
    model, meta = load_latest_model(cfg)

    df = pl.read_parquet(cfg.paths.resolve(cfg.paths.processed_dir) / "features.parquet")
    test = df.filter(pl.col("snapshot_date") == pl.lit(sorted(cfg.snapshots)[-1]).cast(pl.Date))
    sample = test.sample(n=min(sample_n, test.height), seed=cfg.random_seed)
    X = sample.select(feats).to_numpy().astype(float)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = _normalize_shap(explainer.shap_values(X))

    # Global importance.
    mean_abs = np.abs(sv).mean(axis=0)
    importance = sorted(
        [{"feature": f, "label": HUMAN_NAMES.get(f, f), "mean_abs_shap": round(float(m), 4)}
         for f, m in zip(feats, mean_abs)],
        key=lambda d: -d["mean_abs_shap"],
    )
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    (art / "shap_global_importance.json").write_text(json.dumps(importance, indent=2),
                                                     encoding="utf-8")

    shap.summary_plot(sv, X, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(art / "shap_summary.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Example briefings: highest-risk, lowest-risk, and a recurring customer.
    probs = model.predict_proba(X)[:, 1]
    ids = sample[cfg.schema_.customer_id_col].to_list()
    n_events = sample["n_events"].to_numpy()
    picks = {
        "highest_risk": int(np.argmax(probs)),
        "lowest_risk": int(np.argmin(probs)),
        # A recurring customer nearest the 0.5 boundary -- the "Review"-band case
        # where an explanation is most decision-relevant.
        "recurring_review_case": int(np.where(n_events >= 2)[0][
            np.argmin(np.abs(probs[n_events >= 2] - 0.5))
        ]) if (n_events >= 2).any() else int(np.argmax(probs)),
    }
    briefings, text_blocks = {}, []
    for name, i in picks.items():
        b = explain_row(explainer, X[i], feats, float(probs[i]), cfg)
        briefings[name] = {"customer_id": ids[i], **b}
        text_blocks.append(f"### {name}\n" + format_briefing(ids[i], b))

    (art / "example_briefings.json").write_text(json.dumps(briefings, indent=2),
                                                encoding="utf-8")
    (art / "example_briefings.txt").write_text("\n\n".join(text_blocks), encoding="utf-8")

    logger.info("Top global drivers: %s",
                ", ".join(d["feature"] for d in importance[:6]))
    logger.info("Wrote SHAP summary plot, global importance, and %d example briefings",
                len(briefings))
    return {"global_importance": importance, "example_briefings": briefings}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()


if __name__ == "__main__":
    main()
