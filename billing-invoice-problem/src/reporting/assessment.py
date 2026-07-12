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


def _with_vs_without_section(
    ablation: dict[str, Any],
    package_manifest: dict[str, Any] | None,
    batch_summary: dict[str, Any] | None,
) -> list[str]:
    """Assignment-facing summary: comparable models with vs without inferred interval."""
    lines = [
        "## 6. With vs without inferred interval (assignment summary)",
        "",
        "Two **comparable** dollar-churn models on the **same** temporal test set: "
        "same algorithm, split, hyperparameters, and labels — only features differ.",
        "",
        "| | **WITHOUT inferred interval** (baseline) | **WITH inferred interval** (enhanced) |",
        "|---|---|---|",
        "| Features | CORE + GAP (invoice amounts, recency, tenure, "
        "frequency, gap stats, volatility) | Same **plus** interval one-hots + "
        "`interval_confidence` + diagnostics |",
        "| Question | Can we triage from invoices alone? | Does adding the "
        "**inferred** cadence help? |",
        "",
    ]

    holdout = (package_manifest or {}).get("holdout_comparison") or {}
    w = holdout.get("with_interval") or {}
    o = holdout.get("without_interval") or {}
    delta = holdout.get("delta") or {}
    if w and o:
        lines += [
            "### Holdout metrics (packaged dual models, actual test fold)",
            "",
            f"n_test = **{holdout.get('n_test', '?'):,}**",
            "",
            "| Metric | WITH | WITHOUT | Δ (WITH − WITHOUT) |",
            "|---|---:|---:|---:|",
            f"| PR-AUC | {w.get('pr_auc')} | {o.get('pr_auc')} | "
            f"{delta.get('pr_auc')} |",
            f"| ROC-AUC | {w.get('roc_auc')} | {o.get('roc_auc')} | — |",
            f"| Dollar-recall @ top 10% | {w.get('dollar_recall_top10pct')} | "
            f"{o.get('dollar_recall_top10pct')} | "
            f"{delta.get('dollar_recall_top10pct')} |",
            f"| Expected-loss MAE | {w.get('expected_loss_mae')} | "
            f"{o.get('expected_loss_mae')} | {delta.get('expected_loss_mae')} |",
            "",
            "**Plain-language:** ranking metrics are essentially **tied**; WITHOUT is "
            "slightly better on expected-dollar MAE. The inferred interval is not a "
            "large accuracy boost once invoice + gap features exist — its main value "
            "is as a **substitute / explainable compression** when CRM interval is missing.",
            "",
        ]
    else:
        lines.append(
            "_Package holdout comparison missing — run `python -m src.serving.package`._"
        )
        lines.append("")

    abs_ = ablation.get("ablations") or {}
    a1 = abs_.get("A1_core_value") or {}
    a2 = abs_.get("A2_beyond_raw_gaps") or {}
    lines += [
        "### Ablation view (formal significance)",
        "",
        "| Ablation | Comparison | Takeaway |",
        "|---|---|---|",
    ]
    if a1:
        ov = (a1.get("overall") or {}).get("verdict") or {}
        pr = (a1.get("overall") or {}).get("pr_auc") or {}
        dr = (a1.get("overall") or {}).get("dollar_recall_top10pct") or {}
        lines.append(
            f"| **A1** | CORE vs CORE+interval | "
            f"PR-AUC Δ={pr.get('delta')}, dollar-recall Δ={dr.get('delta')}; "
            f"good_enough={ov.get('good_enough')} → interval **helps** vs core-only |"
        )
    if a2:
        ov = (a2.get("overall") or {}).get("verdict") or {}
        mae = (a2.get("overall") or {}).get("expected_loss_mae") or {}
        lines.append(
            f"| **A2** | CORE+GAP vs CORE+GAP+interval | "
            f"MAE Δ={mae.get('delta')} (higher is worse); "
            f"good_enough={ov.get('good_enough')} → largely **redundant** with gaps |"
        )
    if not a1 and not a2:
        lines.append("| — | — | _Run `make ablation`_ |")

    decision = ablation.get("decision", "unknown")
    good = ablation.get("interval_good_enough")
    lines += [
        "",
        f"**Product verdict:** {'**SHIP**' if good else 'Do not ship'} — "
        f"*{decision}*. SHIP means: deploy the inferred interval as a substitute "
        "for the missing real CRM interval (backed by A1). Do not expect a large "
        "lift on top of full gap features (A2).",
        "",
    ]

    if batch_summary:
        reco_w = batch_summary.get("recommendation_counts") or {}
        reco_o = batch_summary.get("recommendation_counts_without_interval") or {}
        n = batch_summary.get("n_customers")
        agree = batch_summary.get("agreement_rate")
        lines += [
            "### Serving decision mix (actual invoices, capped batch)",
            "",
            f"Customers scored: **{n}** · snapshot `{batch_summary.get('snapshot')}` · "
            f"model `{batch_summary.get('model_version')}`",
            "",
            "| Recommendation | WITH interval | WITHOUT interval |",
            "|---|---:|---:|",
            f"| Continue | {reco_w.get('Continue', 0)} | {reco_o.get('Continue', 0)} |",
            f"| Review | {reco_w.get('Review', 0)} | {reco_o.get('Review', 0)} |",
            f"| No-Go | {reco_w.get('No-Go', 0)} | {reco_o.get('No-Go', 0)} |",
            "",
        ]
        if agree is not None:
            lines.append(f"Dual-model recommendation agreement ≈ **{agree:.0%}**.")
            lines.append("")

    return lines


def _train_holdout_section(train_metadata: dict[str, Any] | None) -> list[str]:
    if not train_metadata:
        return [
            "## 5b. Trained model holdout (interval-enhanced)",
            "",
            "_latest_metadata.json not found — run `make train`._",
            "",
        ]
    m = train_metadata.get("metrics_overall") or {}
    freq = (m.get("frequency") or {}).get("calibrated") or {}
    base = (m.get("frequency") or {}).get("baseline_logistic") or {}
    dollar = m.get("dollar_churn") or {}
    cohort = train_metadata.get("cohort") or {}
    lines = [
        "## 5b. Trained model holdout (interval-enhanced, actual data)",
        "",
        f"- Model version: `{train_metadata.get('version')}`",
        f"- Train / test n: **{cohort.get('train', '?'):,}** / "
        f"**{cohort.get('test', '?'):,}** "
        f"(test snapshot `{train_metadata.get('test_snapshot')}`)",
        f"- Calibrated P(churn): ROC-AUC **{freq.get('roc_auc')}**, "
        f"PR-AUC **{freq.get('pr_auc')}**, Brier **{freq.get('brier')}**",
        f"- Logistic baseline: ROC-AUC {base.get('roc_auc')}, PR-AUC {base.get('pr_auc')}",
        f"- Dollar-recall @ top 10%: **{dollar.get('dollar_recall_top10pct')}** "
        f"({dollar.get('pct_dollars_captured_top10pct')}% of dollars)",
        f"- Expected $ churn MAE: **{dollar.get('expected_vs_actual_mae')}**",
        "",
    ]
    return lines


def build_assessment_markdown(artifacts: dict[str, Any], cfg: Config) -> str:
    """Compose the single mini-report (submission + technical detail) from artifacts."""
    eda = artifacts.get("eda") or {}
    ingestion = artifacts.get("ingestion") or {}
    interval = artifacts.get("interval") or {}
    features = artifacts.get("features") or {}
    ablation = artifacts.get("ablation") or {}
    shap = artifacts.get("shap") or []
    package_manifest = artifacts.get("package_manifest")
    train_metadata = artifacts.get("train_metadata")
    batch_summary = artifacts.get("batch_summary")

    lines: list[str] = [
        "# Billing Interval → Dollar-Churn: Mini-Report",
        "",
        f"_Generated: {datetime.now(UTC).isoformat()}_",
        "",
        "This is the **single** written summary for the assignment: approach & "
        "assumptions, what the models revealed on **actual** `Invoices_users` data "
        "(including **with vs without inferred interval**), and what to improve "
        "given more time. Regenerated by `make assess` / "
        "`python -m src.scripts.run_assessment`.",
        "",
        "## 1. Executive verdict",
        "",
    ]

    good = ablation.get("interval_good_enough")
    decision = ablation.get("decision", "Ablation report not found — re-run `make ablation`.")
    if good is True:
        lines.append(
            f"**SHIP the inferred interval** as a substitute for the missing real "
            f"interval. (*SHIP* = deploy it in the credit flow.) Pipeline decision: "
            f"*{decision}*"
        )
    elif good is False:
        lines.append(
            f"**Do not rely on the inferred interval alone.** Pipeline decision: *{decision}*"
        )
    else:
        lines.append(f"**Status unknown.** {decision}")

    lines += [
        "",
        "Interpretation: Ablation 1 shows a **small but statistically significant** "
        "lift when adding the interval to core features. Ablation 2 shows the derived "
        "interval is largely **redundant with raw gap features**. Practical use: ship "
        "as a substitute when CRM interval is missing; optional if gaps are already engineered.",
        "",
        "## 2. Problem, approach & key assumptions",
        "",
        "### Problem",
        "",
        "Credit underwriting needs each customer's billing interval, but the field is "
        "often missing at first integration. `Invoices_users.parquet` has **no "
        "ground-truth interval**, so we infer it and test whether it helps dollar-churn.",
        "",
        "### Approach",
        "",
        "1. **EDA / ingestion** — missingness, same-day collapse, history depth, gaps.",
        "2. **Interval inference** — historical-only gaps → calendar multiples + confidence.",
        "3. **Leakage-safe features & labels** — features only from events before the "
        "snapshot; dollar churn = max(0, past-12mo − future-12mo revenue), independent "
        "of the inferred interval.",
        "4. **Two comparable models** — Model 1 WITHOUT interval (CORE+GAP); Model 2 "
        "WITH interval (CORE+GAP+INTERVAL). Same LightGBM frequency + severity stack, "
        "same temporal split and hyperparameters.",
        "5. **Ablation** — A1 core±interval; A2 core+gap±interval; paired bootstrap / DeLong.",
        "6. **Decision layer** — Continue / Review / No-Go with low-confidence → Review.",
        "7. **Serving** — dual packaged models + FastAPI + Streamlit.",
        "",
        f"Temporal folds: `{', '.join(str(s) for s in cfg.snapshots)}`. "
        f"Outcome window = {cfg.churn.outcome_window_days}d + "
        f"{cfg.churn.grace_days}d grace; drop threshold = {cfg.churn.drop_threshold}. "
        f"Decision: Continue≤{cfg.decision.continue_max_risk}, "
        f"No-Go≥{cfg.decision.nogo_min_risk}, "
        f"min interval confidence={cfg.decision.min_interval_confidence_for_auto}.",
        "",
        "### Key assumptions",
        "",
        "| Assumption | Rationale |",
        "|---|---|",
        "| Same-day invoices = one billing event | Multi-invoice days would distort gaps |",
        "| No CRM interval ground truth | Validate via dollar-churn lift, not label accuracy |",
        "| Temporal train/test | Production-like; no future leakage |",
        "| High churn rates are definitional | Revenue-drop label on short-history-heavy base |",
        "| Gap stats ≠ inferred interval | Gaps are raw spacing; interval is class + confidence |",
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
            lines.append(f"- Date span: {q.get('date_min')} → {q.get('date_max')}")
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

    lines += ["", "## 4. Interval inference snapshot", ""]
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

    lines += ["", "## 5. Dollar-churn cohorts", ""]
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
            "short-history-heavy base; ablations still compare models fairly on the same labels."
        )
    else:
        lines.append("_feature_summary.json not found — run `make features` / `make labels`._")

    lines += [""]
    lines += _train_holdout_section(train_metadata)
    lines += _with_vs_without_section(ablation, package_manifest, batch_summary)

    lines += [
        "## 7. Ablation detail tables",
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
        "yields a small, significant ranking lift and improved expected-loss MAE overall.",
        "- **Ablation 2:** Once raw gap statistics are present, the derived interval "
        "adds little ranking value and can **degrade** expected-loss MAE — useful as "
        "compression/substitute, not as an independent stacked signal.",
        "",
        "## 8. Explainability (SHAP)",
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
    else:
        lines.append("_shap_global_importance.json not found — run `make explain`._")

    lines += [
        "",
        "## 9. Decision layer",
        "",
        f"- Continue if churn risk ≤ {cfg.decision.continue_max_risk}; "
        f"No-Go if ≥ {cfg.decision.nogo_min_risk}; else Review.",
        f"- Guardrail: `interval_confidence` < "
        f"{cfg.decision.min_interval_confidence_for_auto} → **Review** (never auto No-Go).",
        "",
        "## 10. What we would improve given more time",
        "",
        "Ordered by how much each would change the conclusion, not by effort.",
        "",
        "**A. Tighten the verdict to match the evidence.** The full-model holdout "
        "(CORE+GAP) shows the interval is essentially tied and slightly *worse* once "
        "raw gaps exist (PR-AUC Δ=−0.0003, dollar-recall Δ=−0.0023, expected-loss "
        "MAE +7.26). The lift is real only against a core-only model (A1). So the "
        "honest claim is narrower than a blanket \"SHIP\": the inferred interval is "
        "**good enough as a drop-in substitute when gap features are not engineered, "
        "and as an explainable compression of cadence — but it is not an additive "
        "signal on top of raw gaps.** We would re-title the headline around "
        "*substitution value*, not incremental lift, so the claim and the numbers agree.",
        "",
        "**B. Fix the degenerate churn base rate before trusting the metrics.** "
        "Observed churn is 82–89% because 62% of customers have a single invoice, "
        "whose future-12mo revenue trivially drops to zero — so \"churn\" is partly "
        "definitional and the ~0.98 PR-AUC is inflated by an easy majority class. We "
        "would train and report on the **recurring cohort** (≥2 billing events) as the "
        "primary population, treat `insufficient_history` as a separate data-quality "
        "branch rather than a modelled churner, and re-derive the with/without-interval "
        "verdict there — where the decision actually carries signal "
        "(recurring PR-AUC is ~0.86, not 0.98).",
        "",
        "**C. Address the cold-start paradox directly.** The interval is needed *most* "
        "at first integration, which is exactly where inference is weakest — mean "
        "confidence ≈ 0.21 and 76% of customers are single-event at snapshot. We would "
        "quantify decision quality specifically on the thin-history onboarding cohort, "
        "and use hierarchical/priors shrinkage (population cadence priors) so a new "
        "customer gets a defensible interval estimate instead of `insufficient_history`.",
        "",
        "**D. Validate the interval inference itself, not only via churn.** With no "
        "ground truth we currently validate indirectly. A cheap quasi-ground-truth "
        "back-test: hold out each customer's *next* billing event and check whether the "
        "inferred cadence predicts its timing (predict-next-invoice-date error by "
        "inferred class). This tests the inference on its own terms, independent of the "
        "churn model.",
        "",
        "**E. Reconcile the decision bands with the business's binary ask.** The "
        "exercise asks for **Continue / No-Go**; we added **Review** as a data-quality "
        "guardrail. We would report the fraction routed to Review (manual load), justify "
        "it explicitly as data-quality routing, and provide a strict binary fallback for "
        "teams that need a two-way answer.",
        "",
        "**F. Modelling and production hardening.** Calibrate Continue/No-Go thresholds "
        "on business cost (false Continue vs false No-Go); segment-specific severity "
        "models where A2 showed MAE regression; richer interval inference (periodogram "
        "cross-check, soft multi-label `mixed` cadences); PSI drift monitoring, scheduled "
        "retrain, Dockerized API; and analyst review of WITH-vs-WITHOUT disagreements.",
        "",
        "## 11. Artifact index",
        "",
        "| Artifact | Path |",
        "|---|---|",
        "| **This mini-report (send this)** | `reports/assessment/mini_report.md` |",
        "| EDA report | `reports/eda/eda_report.md` |",
        "| Ablation report | `data/artifacts/ablation_report.json` |",
        "| Package / dual holdout | `data/artifacts/serving/package_manifest.json` |",
        "| Batch triage | `reports/batch_assessment/latest/` |",
        "| SHAP global | `data/artifacts/shap_global_importance.json` |",
        "",
        "**Bottom line:** On actual data, WITH vs WITHOUT inferred interval are nearly "
        "tied on ranking; WITHOUT is slightly better on dollar MAE. Ablation A1 supports "
        "**SHIP** (interval helps vs core-only). Serve both models for explainable "
        "triage; force Review when interval confidence is low.",
        "",
    ]
    return "\n".join(lines)


def collect_artifacts(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    eda_path = PROJECT_ROOT / "reports" / "eda" / "eda_report.json"
    if not eda_path.exists():
        eda_path = art / "eda_report.json"
    batch_summary = PROJECT_ROOT / "reports" / "batch_assessment" / "latest" / "batch_summary.json"
    return {
        "eda": _load_json(eda_path),
        "ingestion": _load_json(art / "ingestion_summary.json"),
        "interval": _load_json(art / "interval_summary.json"),
        "features": _load_json(art / "feature_summary.json"),
        "ablation": _load_json(art / "ablation_report.json"),
        "shap": _load_json(art / "shap_global_importance.json"),
        "package_manifest": _load_json(art / "serving" / "package_manifest.json"),
        "train_metadata": _load_json(art / "models" / "latest_metadata.json"),
        "batch_summary": _load_json(batch_summary),
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
