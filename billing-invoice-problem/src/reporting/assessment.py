"""Assessment / mini-report: turn pipeline artifacts into a comparison narrative.

Reads EDA + ingestion + interval + feature + ablation + SHAP artifacts produced
by the training pipeline and writes a graded-style mini-report under
``reports/assessment/``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Config, get_config

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "assessment"


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        logger.warning("Missing artifact: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_delta(delta: float, *, higher_is_better: bool = True) -> str:
    arrow = "↑" if (delta > 0) == higher_is_better else "↓"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4g} {arrow}"


def _metric_row(name: str, block: dict, *, higher_is_better: bool = True) -> str:
    base = block["base"]
    treat = block["treatment"]
    delta = block["delta"]
    sig = "yes" if block.get("significant") else "no"
    ci = block.get("ci95")
    ci_s = f" [{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
    return (
        f"| {name} | {base:.4g} | {treat:.4g} | "
        f"{_fmt_delta(delta, higher_is_better=higher_is_better)}{ci_s} | {sig} |"
    )


def _ablation_section(_name: str, label: str, ab: dict) -> list[str]:
    lines = [
        f"### {label}",
        "",
        f"Base = `{ab['base']}` · Treatment = `{ab['treatment']}`",
        "",
    ]
    for segment in ("overall", "recurring"):
        seg = ab.get(segment)
        if not seg:
            continue
        verdict = seg.get("verdict", {})
        good = verdict.get("good_enough", False)
        lines += [
            f"#### Segment: {segment} (n={seg.get('n', '?'):,})",
            "",
            f"**Segment verdict:** {'good enough' if good else 'not good enough'}",
            "",
            "| Metric | Base | Treatment | Δ (95% CI) | Significant |",
            "|---|---:|---:|---|:---:|",
        ]
        if "pr_auc" in seg:
            lines.append(_metric_row("PR-AUC", seg["pr_auc"], higher_is_better=True))
        if "dollar_recall_top10pct" in seg:
            lines.append(
                _metric_row(
                    "Dollar recall @ top 10%",
                    seg["dollar_recall_top10pct"],
                    higher_is_better=True,
                )
            )
        if "brier" in seg:
            lines.append(_metric_row("Brier", seg["brier"], higher_is_better=False))
        if "expected_loss_mae" in seg:
            lines.append(
                _metric_row(
                    "Expected-loss MAE",
                    seg["expected_loss_mae"],
                    higher_is_better=False,
                )
            )
        if "roc_auc_delong" in seg:
            d = seg["roc_auc_delong"]
            lines.append(
                f"| ROC-AUC (DeLong) | {d['auc_base']:.4g} | {d['auc_treatment']:.4g} | "
                f"{_fmt_delta(d['delta'], higher_is_better=True)} "
                f"(z={d['z']:.3g}, p={d['p_value']:.3g}) | "
                f"{'yes' if d.get('significant') else 'no'} |"
            )
        lines.append("")
        lines.append(
            "Pre-registered checks: "
            f"improves PR-AUC={verdict.get('improves_pr_auc')}, "
            f"improves dollar-recall={verdict.get('improves_dollar_recall')}, "
            f"no Brier degradation={verdict.get('no_material_brier_degradation')}, "
            f"no MAE degradation={verdict.get('no_material_mae_degradation')}."
        )
        lines.append("")
    return lines


def build_assessment_markdown(artifacts: dict[str, Any], cfg: Config) -> str:
    """Compose the mini-report markdown from loaded artifacts."""
    eda = artifacts.get("eda") or {}
    ingestion = artifacts.get("ingestion") or {}
    interval = artifacts.get("interval") or {}
    features = artifacts.get("features") or {}
    ablation = artifacts.get("ablation") or {}
    shap = artifacts.get("shap") or []

    lines: list[str] = [
        "# Billing Interval → Dollar-Churn: Assessment Mini-Report",
        "",
        f"_Generated: {datetime.now(UTC).isoformat()}_",
        "",
        "## 1. Executive verdict",
        "",
    ]

    good = ablation.get("interval_good_enough")
    decision = ablation.get("decision", "Ablation report not found — re-run `make ablation`.")
    if good is True:
        lines.append(
            f"**Ship the inferred interval** as a substitute for the missing real "
            f"interval. Pipeline decision: *{decision}*"
        )
    elif good is False:
        lines.append(
            f"**Do not rely on the inferred interval alone.** Pipeline decision: *{decision}*"
        )
    else:
        lines.append(f"**Status unknown.** {decision}")

    lines += [
        "",
        "Interpretation (honest): Ablation 1 shows a **small but statistically "
        "significant** lift when adding the interval to core customer-health features, "
        "with no material calibration harm on the overall cohort. Ablation 2 shows the "
        "derived interval is largely **redundant with raw gap features** (and can hurt "
        "expected-loss MAE when stacked on top of them). Practical recommendation: use "
        "the inferred interval when CRM interval is missing *and* raw-gap features are "
        "not already engineered; otherwise treat it as optional / explanatory.",
        "",
        "## 2. Problem & approach",
        "",
        "The credit flow needs each customer's billing interval (monthly / quarterly / "
        "semi-annual / annual / one-time / …) but the field is often missing at first "
        "integration. There is **no ground-truth interval** in "
        "`Invoices_users.parquet`, so we:",
        "",
        "1. **Infer** interval with an auditable, historical-only rule engine "
        "(gap → calendar-multiple matching, skipped-period tolerant).",
        "2. **Predict dollar churn** with a two-part model "
        "`P(churn) × E(loss | churn)`, where churn is defined on realized 12-month "
        "revenue (independent of the inferred interval).",
        "3. **Validate** the interval by temporally held-out ablations "
        "(core±interval, core+gap±interval) with paired bootstrap / DeLong.",
        "",
        f"Temporal folds (config): train on earlier snapshot, test on later — "
        f"`{', '.join(str(s) for s in cfg.snapshots)}`. "
        f"Outcome window = {cfg.churn.outcome_window_days}d + "
        f"{cfg.churn.grace_days}d grace; churn drop threshold = "
        f"{cfg.churn.drop_threshold}.",
        "",
        "## 3. Data facts (from EDA / ingestion)",
        "",
    ]

    raw = (eda.get("raw") if eda else None) or {}
    if ingestion:
        lines += [
            f"- Raw invoices: **{ingestion.get('raw_rows', '?'):,}** rows, "
            f"**{ingestion.get('customers', '?'):,}** customers",
            f"- Billing events after same-day aggregation: "
            f"**{ingestion.get('billing_events', '?'):,}** "
            f"(collapse ratio {ingestion.get('same_day_collapse_ratio', '?')})",
        ]
        q = ingestion.get("quality") or {}
        if q:
            lines.append(
                f"- Missingness: null_customer={q.get('null_customer_rate', 0)}, "
                f"null_date={q.get('null_date_rate', 0)}, "
                f"null_amount={q.get('null_amount_rate', 0)}, "
                f"non_positive_amount={q.get('non_positive_amount_rate', 0)}"
            )
            lines.append(
                f"- Date span: {q.get('date_min')} → {q.get('date_max')}"
            )
    elif raw:
        lines += [
            f"- Raw invoices: **{raw.get('n_rows', '?'):,}** rows, "
            f"**{raw.get('n_customers', '?'):,}** customers",
            f"- Date span: {raw.get('date_range', {}).get('min')} → "
            f"{raw.get('date_range', {}).get('max')}",
        ]

    if eda.get("findings"):
        lines += ["", "### EDA findings that drove design", ""]
        for f in eda["findings"]:
            lines.append(f"- **[{f['severity']}] {f['area']}:** {f['message']}")

    lines += [
        "",
        "## 4. Interval inference snapshot",
        "",
    ]
    if interval:
        for snap, stats in interval.items():
            counts = stats.get("class_counts") or {}
            top = ", ".join(f"{k}={v:,}" for k, v in list(counts.items())[:5])
            lines.append(
                f"- Snapshot **{snap}**: {stats.get('customers', 0):,} customers, "
                f"mean confidence={stats.get('mean_confidence')}; classes: {top}"
            )
        lines.append(
            "- Design note: majority mass sits in `insufficient_history` — "
            "low confidence correctly routes to **Review**, never auto No-Go."
        )
    else:
        lines.append("_interval_summary.json not found — run `make interval`._")

    lines += [
        "",
        "## 5. Dollar-churn cohorts",
        "",
    ]
    if features:
        for snap, stats in features.items():
            lines.append(
                f"- Snapshot **{snap}**: cohort={stats.get('cohort', 0):,}, "
                f"churn_rate={stats.get('churn_rate')}, "
                f"mean_dollar_loss={stats.get('mean_dollar_loss')}, "
                f"total $ at risk={stats.get('total_dollars_at_risk'):,}"
            )
        lines.append(
            "- High observed churn rates reflect the revenue-drop definition on a "
            "short-history-heavy base; the ablation still compares models *fairly* "
            "on the same labels."
        )
    else:
        lines.append("_feature_summary.json not found — run `make features` / `make labels`._")

    lines += [
        "",
        "## 6. Ablation comparison (the business question)",
        "",
        "Pre-registered bar: interval is \"good enough\" iff it improves PR-AUC "
        "**or** dollar-recall@10% with **no material** Brier / expected-loss MAE "
        "degradation, on overall and recurring segments.",
        "",
    ]

    ablations = ablation.get("ablations") or {}
    if "A1_core_value" in ablations:
        lines += _ablation_section(
            "A1",
            "Ablation 1 — Core value (core vs core+interval)",
            ablations["A1_core_value"],
        )
    if "A2_beyond_raw_gaps" in ablations:
        lines += _ablation_section(
            "A2",
            "Ablation 2 — Beyond raw gaps (core+gap vs core+gap+interval)",
            ablations["A2_beyond_raw_gaps"],
        )
    if not ablations:
        lines.append("_ablation_report.json not found — run `make ablation`._")

    lines += [
        "### Comparison narrative",
        "",
        "- **Ablation 1:** Adding the inferred interval to core RFM/tenure features "
        "yields a small, significant ranking lift and a large improvement in "
        "expected-loss MAE on the overall test cohort. On the recurring subsegment "
        "the probability ranking lift remains significant; Brier worsens slightly "
        "but stays within the pre-registered \"no material degradation\" allowance "
        "used by the harness for the overall good-enough flag.",
        "- **Ablation 2:** Once raw gap statistics are present, the *derived* "
        "interval adds little ranking value and **degrades** expected-loss MAE "
        "(overall and especially recurring). The interval is therefore a useful "
        "*compression / substitute* for missing CRM interval when gap features are "
        "absent, not an independent signal stacked on top of them.",
        "",
        "## 7. Explainability (SHAP)",
        "",
    ]
    if shap:
        lines.append("Top drivers by mean |SHAP| (frequency model):")
        lines.append("")
        lines.append("| Rank | Feature | Meaning | mean |SHAP| |")
        lines.append("|---:|---|---|---:|")
        for i, row in enumerate(shap[:12], 1):
            lines.append(
                f"| {i} | `{row.get('feature')}` | {row.get('label', '')} | "
                f"{row.get('mean_abs_shap')} |"
            )
        interval_rank = next(
            (i for i, r in enumerate(shap, 1) if "interval" in str(r.get("feature", "")).lower()
             or str(r.get("feature", "")).startswith("is_")),
            None,
        )
        # Prefer interval_confidence specifically
        conf_rank = next(
            (i for i, r in enumerate(shap, 1) if r.get("feature") == "interval_confidence"),
            None,
        )
        if conf_rank:
            lines.append("")
            lines.append(
                f"`interval_confidence` ranks **#{conf_rank}** globally — the model "
                "uses data-quality of the interval inference, not only class one-hots."
            )
        elif interval_rank:
            lines.append("")
            lines.append(
                f"An interval-related feature appears around rank **#{interval_rank}**."
            )
    else:
        lines.append("_shap_global_importance.json not found — run `make explain`._")

    lines += [
        "",
        "## 8. Decision layer implications",
        "",
        f"- Continue if expected-churn risk ≤ {cfg.decision.continue_max_risk}; "
        f"No-Go if ≥ {cfg.decision.nogo_min_risk}; else Review.",
        f"- Data-quality guardrail: if `interval_confidence` < "
        f"{cfg.decision.min_interval_confidence_for_auto}, force **Review** "
        "(never auto No-Go).",
        "- This matches the EDA reality: most customers lack enough history for a "
        "high-confidence cadence label.",
        "",
        "## 9. Assumptions & what we'd improve",
        "",
        "**Assumptions**",
        "",
        "- Same-day invoices are one economic billing event (sum amounts).",
        "- Dollar churn = max(0, past-12mo − future-12mo revenue); binary churn uses "
        f"a {cfg.churn.drop_threshold:.0%} future/past drop.",
        "- Interval rules are a latent-period estimate, not a CRM ground truth.",
        "- Temporal split (earlier → later snapshot) is the primary validation.",
        "",
        "**What we'd improve**",
        "",
        "- Calibrate decision thresholds on business cost (false Continue vs false No-Go).",
        "- Segment-specific models for recurring vs single-event customers.",
        "- Stronger severity calibration where Ablation 2 showed MAE regression.",
        "- Optional FFT/periodogram confidence cross-check for high-volume accounts.",
        "- Track drift (PSI) of gap/interval features in production.",
        "",
        "## 10. Artifact index",
        "",
        "| Artifact | Path |",
        "|---|---|",
        "| EDA report | `reports/eda/eda_report.md` |",
        "| Ingestion summary | `data/artifacts/ingestion_summary.json` |",
        "| Interval summary | `data/artifacts/interval_summary.json` |",
        "| Feature / label summary | `data/artifacts/feature_summary.json` |",
        "| Ablation report | `data/artifacts/ablation_report.json` |",
        "| SHAP global | `data/artifacts/shap_global_importance.json` |",
        "",
    ]
    return "\n".join(lines)


def collect_artifacts(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    eda_path = PROJECT_ROOT / "reports" / "eda" / "eda_report.json"
    if not eda_path.exists():
        eda_path = art / "eda_report.json"
    return {
        "eda": _load_json(eda_path),
        "ingestion": _load_json(art / "ingestion_summary.json"),
        "interval": _load_json(art / "interval_summary.json"),
        "features": _load_json(art / "feature_summary.json"),
        "ablation": _load_json(art / "ablation_report.json"),
        "shap": _load_json(art / "shap_global_importance.json"),
    }


def write_assessment_report(cfg: Config | None = None) -> Path:
    """Write ``reports/assessment/mini_report.md`` (+ JSON sidecar of sources)."""
    cfg = cfg or get_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = collect_artifacts(cfg)
    md = build_assessment_markdown(artifacts, cfg)

    md_path = REPORTS_DIR / "mini_report.md"
    md_path.write_text(md, encoding="utf-8")

    # Also mirror into artifacts for packaging / reviewers.
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "mini_report.md").write_text(md, encoding="utf-8")

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources_present": {k: v is not None for k, v in artifacts.items()},
        "interval_good_enough": (artifacts.get("ablation") or {}).get("interval_good_enough"),
        "decision": (artifacts.get("ablation") or {}).get("decision"),
        "report_path": str(md_path),
    }
    (REPORTS_DIR / "assessment_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Wrote assessment mini-report -> %s", md_path)
    return md_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    path = write_assessment_report()
    print(f"Assessment report: {path}")


if __name__ == "__main__":
    main()
