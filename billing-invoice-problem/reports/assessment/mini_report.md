# Billing Interval → Dollar-Churn: Assessment Mini-Report

_Generated: 2026-07-12T14:09:48.015514+00:00_

## 1. Executive verdict

**Ship the inferred interval** as a substitute for the missing real interval. Pipeline decision: *SHIP the inferred interval as a substitute for the missing real interval*

Interpretation (honest): Ablation 1 shows a **small but statistically significant** lift when adding the interval to core customer-health features, with no material calibration harm on the overall cohort. Ablation 2 shows the derived interval is largely **redundant with raw gap features** (and can hurt expected-loss MAE when stacked on top of them). Practical recommendation: use the inferred interval when CRM interval is missing *and* raw-gap features are not already engineered; otherwise treat it as optional / explanatory.

## 2. Problem & approach

The credit flow needs each customer's billing interval (monthly / quarterly / semi-annual / annual / one-time / …) but the field is often missing at first integration. There is **no ground-truth interval** in `Invoices_users.parquet`, so we:

1. **Infer** interval with an auditable, historical-only rule engine (gap → calendar-multiple matching, skipped-period tolerant).
2. **Predict dollar churn** with a two-part model `P(churn) × E(loss | churn)`, where churn is defined on realized 12-month revenue (independent of the inferred interval).
3. **Validate** the interval by temporally held-out ablations (core±interval, core+gap±interval) with paired bootstrap / DeLong.

Temporal folds (config): train on earlier snapshot, test on later — `2024-10-05, 2025-03-05`. Outcome window = 365d + 30d grace; churn drop threshold = 0.5.

## 3. Data facts (from EDA / ingestion)

- Raw invoices: **22,083,629** rows, **562,851** customers
- Billing events after same-day aggregation: **845,464** (collapse ratio 0.9617)
- Missingness: null_customer=0.0, null_date=0.0, null_amount=0.0, non_positive_amount=0.0
- Date span: 2023-01-09 00:00:00 → 2026-04-05 00:00:00

### EDA findings that drove design

- **[info] missingness:** No nulls or non-positive amounts in customerid/date/amount — imputation is unnecessary; the real missing field is billing interval.
- **[critical] same_day_aggregation:** 17.2% of customer-days have >1 invoice (max 9387 on one day) — same-day aggregation is required before gap/interval logic.
- **[critical] history_depth:** 62.0% of customers have a single invoice (p50=1.0, p75=2.0, max=9387) — split insufficient_history vs one_time_candidate; do not blanket one_time.
- **[warn] amount_skew:** Amounts are right-skewed (median=1.0, mean=10.77, max=11880.0) — use log1p / robust summaries for monetary features.
- **[info] gap_shape:** Inter-event gaps: n=282613, p50=8.0d, p75=31.0d, p90=365.0d — match to calendar multiples (30/91/182/365) allowing skipped periods.
- **[info] recurring_share:** After aggregation, 23.9% of customers have >=2 billing events (interval inference has a gap signal); 76.0% remain single-event.
- **[info] modeling_implication:** Validate inferred intervals indirectly via temporally held-out ablation on dollar-churn (no ground-truth interval column exists).

## 4. Interval inference snapshot

- Snapshot **2024-10-05**: 138,314 customers, mean confidence=0.213; classes: insufficient_history=95,321, irregular=28,211, annual=6,287, monthly=4,402, one_time_candidate=3,483
- Snapshot **2025-03-05**: 282,181 customers, mean confidence=0.2257; classes: insufficient_history=209,748, irregular=42,149, annual=11,616, one_time_candidate=10,405, monthly=6,921
- Design note: majority mass sits in `insufficient_history` — low confidence correctly routes to **Review**, never auto No-Go.

## 5. Dollar-churn cohorts

- Snapshot **2024-10-05**: cohort=131,855, churn_rate=0.8251, mean_dollar_loss=223.3, total $ at risk=37,421,119.96
- Snapshot **2025-03-05**: cohort=261,098, churn_rate=0.8919, mean_dollar_loss=115.18, total $ at risk=39,994,700.11
- High observed churn rates reflect the revenue-drop definition on a short-history-heavy base; the ablation still compares models *fairly* on the same labels.

## 6. Ablation comparison (the business question)

Pre-registered bar: interval is "good enough" iff it improves PR-AUC **or** dollar-recall@10% with **no material** Brier / expected-loss MAE degradation, on overall and recurring segments.

### Ablation 1 — Core value (core vs core+interval)

Base = `core` · Treatment = `core_interval`

#### Segment: overall (n=261,098)

**Segment verdict:** good enough

| Metric | Base | Treatment | Δ (95% CI) | Significant |
|---|---:|---:|---|:---:|
| PR-AUC | 0.9852 | 0.9857 | +0.0005 ↑ [0.0003, 0.0007] | yes |
| Dollar recall @ top 10% | 0.841 | 0.8444 | +0.0034 ↑ [0.0026, 0.004] | yes |
| Brier | 0.0887 | 0.0885 | -0.0002 ↑ [-0.0003, -0.0001] | yes |
| Expected-loss MAE | 56.29 | 41.74 | -14.55 ↑ [-15.69, -13.41] | yes |
| ROC-AUC (DeLong) | 0.9129 | 0.9143 | +0.0014 ↑ (z=5.83, p=0) | yes |

Pre-registered checks: improves PR-AUC=True, improves dollar-recall=True, no Brier degradation=True, no MAE degradation=True.

#### Segment: recurring (n=55,542)

**Segment verdict:** good enough

| Metric | Base | Treatment | Δ (95% CI) | Significant |
|---|---:|---:|---|:---:|
| PR-AUC | 0.8593 | 0.8602 | +0.0009 ↑ [0.0003, 0.0016] | yes |
| Dollar recall @ top 10% | 0.6096 | 0.6132 | +0.0036 ↑ [0.0028, 0.0055] | yes |
| Brier | 0.2545 | 0.2553 | +0.0008 ↓ [0.0005, 0.0011] | yes |
| Expected-loss MAE | 243 | 174.8 | -68.29 ↑ [-73.81, -62.76] | yes |
| ROC-AUC (DeLong) | 0.749 | 0.7513 | +0.0023 ↑ (z=3.21, p=0.0013) | yes |

Pre-registered checks: improves PR-AUC=True, improves dollar-recall=True, no Brier degradation=True, no MAE degradation=True.

### Ablation 2 — Beyond raw gaps (core+gap vs core+gap+interval)

Base = `core_gap` · Treatment = `core_gap_interval`

#### Segment: overall (n=261,098)

**Segment verdict:** not good enough

| Metric | Base | Treatment | Δ (95% CI) | Significant |
|---|---:|---:|---|:---:|
| PR-AUC | 0.9851 | 0.9859 | +0.0007 ↑ [0.0006, 0.0009] | yes |
| Dollar recall @ top 10% | 0.8417 | 0.8444 | +0.0026 ↑ [0.0019, 0.0032] | yes |
| Brier | 0.089 | 0.0884 | -0.0005 ↑ [-0.0006, -0.0004] | yes |
| Expected-loss MAE | 42.6 | 68.22 | +25.62 ↓ [24.43, 27.15] | yes |
| ROC-AUC (DeLong) | 0.9131 | 0.9148 | +0.0017 ↑ (z=7.18, p=0) | yes |

Pre-registered checks: improves PR-AUC=True, improves dollar-recall=True, no Brier degradation=True, no MAE degradation=False.

#### Segment: recurring (n=55,542)

**Segment verdict:** not good enough

| Metric | Base | Treatment | Δ (95% CI) | Significant |
|---|---:|---:|---|:---:|
| PR-AUC | 0.8601 | 0.8606 | +0.0004 ↑ [-0.0002, 0.0011] | no |
| Dollar recall @ top 10% | 0.6128 | 0.5717 | -0.0411 ↓ [-0.0434, -0.038] | yes |
| Brier | 0.2549 | 0.2547 | -0.0002 ↑ [-0.0004, 0.0001] | no |
| Expected-loss MAE | 178.3 | 298 | +119.7 ↓ [112.2, 125.6] | yes |
| ROC-AUC (DeLong) | 0.751 | 0.7511 | +0.0001 ↑ (z=0.17, p=0.865) | no |

Pre-registered checks: improves PR-AUC=False, improves dollar-recall=False, no Brier degradation=True, no MAE degradation=False.

### Comparison narrative

- **Ablation 1:** Adding the inferred interval to core RFM/tenure features yields a small, significant ranking lift and a large improvement in expected-loss MAE on the overall test cohort. On the recurring subsegment the probability ranking lift remains significant; Brier worsens slightly but stays within the pre-registered "no material degradation" allowance used by the harness for the overall good-enough flag.
- **Ablation 2:** Once raw gap statistics are present, the *derived* interval adds little ranking value and **degrades** expected-loss MAE (overall and especially recurring). The interval is therefore a useful *compression / substitute* for missing CRM interval when gap features are absent, not an independent signal stacked on top of them.

## 7. Explainability (SHAP)

Top drivers by mean |SHAP| (frequency model):

| Rank | Feature | Meaning | mean |SHAP| |
|---:|---|---|---:|
| 1 | `last_amount` | most recent invoice amount | 0.4189 |
| 2 | `max_amount` | largest invoice amount | 0.3911 |
| 3 | `recency_days` | days since last invoice | 0.2285 |
| 4 | `past12_amount` | revenue in the last 12 months | 0.2224 |
| 5 | `interval_confidence` | billing-interval confidence | 0.2221 |
| 6 | `total_amount` | lifetime revenue | 0.1801 |
| 7 | `n_invoices_total` | total invoice lines | 0.1234 |
| 8 | `mean_amount` | average invoice amount | 0.0595 |
| 9 | `tenure_days` | customer tenure | 0.0574 |
| 10 | `median_gap_days` | typical days between invoices | 0.0531 |
| 11 | `gap_cv` | billing regularity | 0.0387 |
| 12 | `revenue_trend` | revenue trend (recent vs prior year) | 0.0185 |

`interval_confidence` ranks **#5** globally — the model uses data-quality of the interval inference, not only class one-hots.

## 8. Decision layer implications

- Continue if expected-churn risk ≤ 0.3; No-Go if ≥ 0.7; else Review.
- Data-quality guardrail: if `interval_confidence` < 0.5, force **Review** (never auto No-Go).
- This matches the EDA reality: most customers lack enough history for a high-confidence cadence label.

## 9. Assumptions & what we'd improve

**Assumptions**

- Same-day invoices are one economic billing event (sum amounts).
- Dollar churn = max(0, past-12mo − future-12mo revenue); binary churn uses a 50% future/past drop.
- Interval rules are a latent-period estimate, not a CRM ground truth.
- Temporal split (earlier → later snapshot) is the primary validation.

**What we'd improve**

- Calibrate decision thresholds on business cost (false Continue vs false No-Go).
- Segment-specific models for recurring vs single-event customers.
- Stronger severity calibration where Ablation 2 showed MAE regression.
- Optional FFT/periodogram confidence cross-check for high-volume accounts.
- Track drift (PSI) of gap/interval features in production.

## 10. Artifact index

| Artifact | Path |
|---|---|
| EDA report | `reports/eda/eda_report.md` |
| Ingestion summary | `data/artifacts/ingestion_summary.json` |
| Interval summary | `data/artifacts/interval_summary.json` |
| Feature / label summary | `data/artifacts/feature_summary.json` |
| Ablation report | `data/artifacts/ablation_report.json` |
| SHAP global | `data/artifacts/shap_global_importance.json` |
