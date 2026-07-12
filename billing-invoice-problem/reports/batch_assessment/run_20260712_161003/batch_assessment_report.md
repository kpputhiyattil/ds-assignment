# Batch Invoice Assessment Report

_Generated: 2026-07-12T16:10:03.912615+00:00_
_Run dir: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\reports\batch_assessment\run_20260712_161003`_

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
- Customers scored: **97**
- Invoice rows loaded: **2873**
- Recurring-only (>=2 billing days): **True**
- Truncated: True (max_customers=200, allow_full=False)
- Model version: `20260712_032913`
- Mean P(churn): 0.6484
- Total expected $ churn: 39,893.52
- Mean interval confidence: 0.3677

### Recommendation mix (config thresholds)

| Recommendation | Count | Share |
|---|---:|---:|
| Continue | 0 | 0.0% |
| Review | 86 | 88.7% |
| No-Go | 11 | 11.3% |

## 1b. Threshold sensitivity

Same scored probabilities/confidences, re-bucketed under alternate decision thresholds. Rows marked `*` use the config defaults (Continue≤0.3, No-Go≥0.7, min_conf=0.5).

| Cont≤ | No-Go≥ | Min conf | Continue % | Review % | No-Go % | * |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.20 | 0.60 | 0.30 | 0.0 | 68.0 | 32.0 |  |
| 0.20 | 0.70 | 0.30 | 0.0 | 72.2 | 27.8 |  |
| 0.20 | 0.80 | 0.30 | 0.0 | 79.4 | 20.6 |  |
| 0.30 | 0.60 | 0.30 | 0.0 | 68.0 | 32.0 |  |
| 0.30 | 0.70 | 0.30 | 0.0 | 72.2 | 27.8 |  |
| 0.30 | 0.80 | 0.30 | 0.0 | 79.4 | 20.6 |  |
| 0.40 | 0.60 | 0.30 | 9.3 | 58.8 | 32.0 |  |
| 0.40 | 0.70 | 0.30 | 9.3 | 62.9 | 27.8 |  |
| 0.40 | 0.80 | 0.30 | 9.3 | 70.1 | 20.6 |  |
| 0.50 | 0.60 | 0.30 | 41.2 | 26.8 | 32.0 |  |
| 0.50 | 0.70 | 0.30 | 41.2 | 30.9 | 27.8 |  |
| 0.50 | 0.80 | 0.30 | 41.2 | 38.1 | 20.6 |  |
| 0.20 | 0.60 | 0.50 | 0.0 | 88.7 | 11.3 |  |
| 0.20 | 0.70 | 0.50 | 0.0 | 88.7 | 11.3 |  |
| 0.20 | 0.80 | 0.50 | 0.0 | 89.7 | 10.3 |  |
| 0.30 | 0.60 | 0.50 | 0.0 | 88.7 | 11.3 |  |
| 0.30 | 0.70 | 0.50 | 0.0 | 88.7 | 11.3 | * |
| 0.30 | 0.80 | 0.50 | 0.0 | 89.7 | 10.3 |  |
| 0.40 | 0.60 | 0.50 | 3.1 | 85.6 | 11.3 |  |
| 0.40 | 0.70 | 0.50 | 3.1 | 85.6 | 11.3 |  |
| 0.40 | 0.80 | 0.50 | 3.1 | 86.6 | 10.3 |  |
| 0.50 | 0.60 | 0.50 | 12.4 | 76.3 | 11.3 |  |
| 0.50 | 0.70 | 0.50 | 12.4 | 76.3 | 11.3 |  |
| 0.50 | 0.80 | 0.50 | 12.4 | 77.3 | 10.3 |  |

Full sweep (all min_conf × continue × nogo) is in `threshold_sensitivity.json` in this run folder.


## 2. Explainable briefings (top risk)

### yESJ9JNkiUisZkcc

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 3479.9 · Interval: `annual` (conf=0.6636)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** revenue in the last 12 months (increasing risk)

Risk ↑:
- revenue in the last 12 months (3,480.00) [shap +2.389]
- total invoice lines (11) [shap +0.837]
- billing regularity (3.08) [shap +0.750]
- lifetime revenue (3,828.00) [shap +0.707]

Risk ↓:
- billing-interval confidence (0.66) [shap -0.436]
- customer tenure (375) [shap -0.052]
- flag: irregular cadence (no) [shap -0.010]
- flag: quarterly cadence (no) [shap -0.008]

### z6IKJB6hKXmDMhQj

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 440.72 · Interval: `monthly` (conf=0.6882)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (34) [shap +2.741]
- days since last invoice (302) [shap +1.373]
- number of billing events (34) [shap +0.596]
- billing events in the last 12 months (11) [shap +0.479]

Risk ↓:
- billing-interval confidence (0.69) [shap -0.478]
- most recent invoice amount (49.00) [shap -0.410]
- customer tenure (457) [shap -0.174]
- largest invoice amount (49.00) [shap -0.139]

### AGVmtCHYn0x7Ss7E

- Recommendation: **Review** (low_interval_confidence)
- P(churn): 1.0 · Expected $: 7491.35 · Interval: `irregular` (conf=0.05)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** revenue in the last 12 months (increasing risk)

Risk ↑:
- revenue in the last 12 months (7,309.00) [shap +2.341]
- total invoice lines (22) [shap +2.108]
- average invoice amount (2,436.33) [shap +0.861]
- lifetime revenue (7,309.00) [shap +0.829]

Risk ↓:
- most recent invoice amount (3,480.00) [shap -0.085]
- billing events in the last 12 months (3) [shap -0.028]
- invoice-amount variability (0.87) [shap -0.015]
- customer tenure (8) [shap -0.011]
