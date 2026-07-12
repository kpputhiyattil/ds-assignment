# Batch Invoice Assessment Report

_Generated: 2026-07-12T16:03:05.431709+00:00_
_Run dir: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\reports\batch_assessment\run_20260712_160305`_

## 0. Assignment checklist — interval value test

Part 3 of the brief (*compare churn model with vs without the inferred interval*) is answered by the **ablation harness** (`make ablation`), not by re-scoring customers. Batch assessment below is the *serving* pass (Continue / Review / No-Go on invoice history).

**Verdict:** `SHIP the inferred interval as a substitute for the missing real interval`
- `interval_good_enough` = **True**
- Source: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\data\artifacts\ablation_report.json`
- Narrative: `reports/assessment/mini_report.md`

| Ablation | Base → Treatment | PR-AUC Δ | Dollar-recall@10% Δ | MAE Δ | Segment good_enough |
|---|---|---:|---:|---:|:---:|
| A1 core value | `core` → `core_interval` | 0.0005 | 0.0034 | -14.5537 | True |
| A2 beyond raw gaps | `core_gap` → `core_gap_interval` | 0.0007 | 0.0026 | 25.618 | False |

- **A1**: core features vs core+interval — tests whether the inferred interval adds value when CRM interval is missing.
- **A2**: core+gap vs core+gap+interval — tests whether the *derived* interval adds anything beyond raw gap stats.

## 1. Serving batch summary

- Snapshot: **2025-03-05**
- Customers scored: **50**
- Invoice rows loaded: **456**
- Truncated: True (max_customers=50, allow_full=False)
- Model version: `20260712_032913`
- Mean P(churn): 0.8773
- Total expected $ churn: 12,842.74
- Mean interval confidence: 0.5741

### Recommendation mix

| Recommendation | Count | Share |
|---|---:|---:|
| Continue | 0 | 0.0% |
| Review | 29 | 58.0% |
| No-Go | 21 | 42.0% |

## 2. Explainable briefings (top risk)

### ZKqukMSyssgcLGsb

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 2063.35 · Interval: `quarterly` (conf=0.6321)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (13) [shap +1.241]
- billing regularity (3.36) [shap +1.161]
- revenue in the last 12 months (1,044.00) [shap +1.043]
- lifetime revenue (8,004.00) [shap +0.894]

Risk ↓:
- largest invoice amount (696.00) [shap -0.398]
- billing-interval confidence (0.63) [shap -0.268]
- most recent invoice amount (348.00) [shap -0.039]
- revenue in the prior 12 months (0.00) [shap -0.038]

### VwwjsRpBjzpemQkD

- Recommendation: **Review** (low_interval_confidence)
- P(churn): 1.0 · Expected $: 3591.93 · Interval: `mixed` (conf=0.2913)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** revenue in the last 12 months (increasing risk)

Risk ↑:
- revenue in the last 12 months (3,480.00) [shap +2.077]
- total invoice lines (13) [shap +1.064]
- lifetime revenue (4,177.00) [shap +0.789]
- billing regularity (2.52) [shap +0.595]

Risk ↓:
- billing-interval confidence (0.29) [shap -0.080]
- competing cadences (2) [shap -0.016]
- flag: irregular cadence (no) [shap -0.005]
- flag: quarterly cadence (no) [shap -0.004]
