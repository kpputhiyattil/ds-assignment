# Batch Invoice Assessment Report

_Generated: 2026-07-12T16:04:10.426032+00:00_
_Run dir: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\reports\batch_assessment\run_20260712_160410`_

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
- Customers scored: **500**
- Invoice rows loaded: **6790**
- Truncated: True (max_customers=500, allow_full=False)
- Model version: `20260712_032913`
- Mean P(churn): 0.9126
- Total expected $ churn: 91,265.32
- Mean interval confidence: 0.5272

### Recommendation mix

| Recommendation | Count | Share |
|---|---:|---:|
| Continue | 0 | 0.0% |
| Review | 256 | 51.2% |
| No-Go | 244 | 48.8% |

## 2. Explainable briefings (top risk)

### AsjiD97K8QyIPNXc

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.92 · Interval: `monthly` (conf=0.587)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (15) [shap +1.728]
- revenue trend (recent vs prior year) (0.00) [shap +0.707]
- days since last invoice (618) [shap +0.706]
- number of billing events (15) [shap +0.499]

Risk ↓:
- billing-interval confidence (0.59) [shap -0.306]
- most recent invoice amount (49.00) [shap -0.216]
- lifetime revenue (735.00) [shap -0.074]
- largest invoice amount (49.00) [shap -0.059]

### 7IMDlQBsCGZvKMdz

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 636.17 · Interval: `monthly` (conf=0.6857)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (28) [shap +2.470]
- days since last invoice (263) [shap +2.046]
- number of billing events (28) [shap +0.790]
- billing regularity (0.80) [shap +0.406]

Risk ↓:
- billing-interval confidence (0.69) [shap -0.380]
- most recent invoice amount (49.00) [shap -0.295]
- revenue in the prior 12 months (686.00) [shap -0.293]
- billing events in the last 12 months (13) [shap -0.218]

### 5WMfBPL6Riq9lJeb

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 30.02 · Interval: `monthly` (conf=0.6131)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (26) [shap +2.818]
- days since last invoice (567) [shap +0.858]
- average invoice amount (49.00) [shap +0.490]
- revenue trend (recent vs prior year) (0.00) [shap +0.446]

Risk ↓:
- billing-interval confidence (0.61) [shap -0.375]
- largest invoice amount (49.00) [shap -0.137]
- most recent invoice amount (49.00) [shap -0.072]
- revenue in the last 12 months (0.00) [shap -0.054]

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

### bgAg16iGtrb7duHn

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.92 · Interval: `monthly` (conf=0.5989)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (15) [shap +1.758]
- days since last invoice (589) [shap +0.779]
- revenue trend (recent vs prior year) (0.00) [shap +0.737]
- number of billing events (15) [shap +0.510]

Risk ↓:
- billing-interval confidence (0.60) [shap -0.299]
- most recent invoice amount (49.00) [shap -0.201]
- largest invoice amount (49.00) [shap -0.068]
- lifetime revenue (735.00) [shap -0.065]

### xlUqhCrP5CrlTzn0

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.92 · Interval: `monthly` (conf=0.587)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (15) [shap +1.728]
- revenue trend (recent vs prior year) (0.00) [shap +0.707]
- days since last invoice (618) [shap +0.706]
- number of billing events (15) [shap +0.499]

Risk ↓:
- billing-interval confidence (0.59) [shap -0.306]
- most recent invoice amount (49.00) [shap -0.216]
- lifetime revenue (735.00) [shap -0.074]
- largest invoice amount (49.00) [shap -0.059]

### lqPSqKe82ruz6JSN

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 30.51 · Interval: `monthly` (conf=0.6862)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (29) [shap +2.815]
- days since last invoice (374) [shap +0.947]
- number of billing events (29) [shap +0.458]
- average invoice amount (49.00) [shap +0.435]

Risk ↓:
- billing-interval confidence (0.69) [shap -0.339]
- customer tenure (399) [shap -0.281]
- largest invoice amount (49.00) [shap -0.139]
- most recent invoice amount (49.00) [shap -0.126]

### UE1WrbgfxSvXjj6p

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 843.49 · Interval: `monthly` (conf=0.6321)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (37) [shap +2.567]
- revenue in the last 12 months (792.00) [shap +0.770]
- days since last invoice (150) [shap +0.582]
- number of billing events (31) [shap +0.416]

Risk ↓:
- customer tenure (608) [shap -0.339]
- billing-interval confidence (0.63) [shap -0.260]
- typical days between invoices (30) [shap -0.206]
- billing events in the last 12 months (18) [shap -0.192]

### jEQaZV7OclV6aAyr

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.39 · Interval: `monthly` (conf=0.5743)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (16) [shap +1.760]
- days since last invoice (634) [shap +0.795]
- revenue trend (recent vs prior year) (0.00) [shap +0.619]
- average invoice amount (49.00) [shap +0.506]

Risk ↓:
- billing-interval confidence (0.57) [shap -0.360]
- most recent invoice amount (49.00) [shap -0.136]
- lifetime revenue (784.00) [shap -0.087]
- largest invoice amount (49.00) [shap -0.063]

### Ztvcmw00hyI8oB7z

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.53 · Interval: `monthly` (conf=0.617)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (13) [shap +1.606]
- days since last invoice (528) [shap +0.937]
- revenue trend (recent vs prior year) (0.00) [shap +0.865]
- billing regularity (0.71) [shap +0.508]

Risk ↓:
- billing-interval confidence (0.62) [shap -0.322]
- most recent invoice amount (49.00) [shap -0.131]
- lifetime revenue (637.00) [shap -0.064]
- dominant-cadence share (1.00) [shap -0.028]

### iVD4y5sawWJT9oeu

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.49 · Interval: `monthly` (conf=0.5724)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (14) [shap +1.746]
- days since last invoice (649) [shap +0.747]
- revenue trend (recent vs prior year) (0.00) [shap +0.640]
- number of billing events (14) [shap +0.483]

Risk ↓:
- billing-interval confidence (0.57) [shap -0.343]
- most recent invoice amount (49.00) [shap -0.163]
- largest invoice amount (49.00) [shap -0.105]
- lifetime revenue (686.00) [shap -0.070]

### QwIPFHjXUlH7AzRR

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.37 · Interval: `monthly` (conf=0.6344)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (14) [shap +1.569]
- revenue trend (recent vs prior year) (0.00) [shap +0.904]
- days since last invoice (483) [shap +0.865]
- number of billing events (14) [shap +0.521]

Risk ↓:
- billing-interval confidence (0.63) [shap -0.305]
- most recent invoice amount (49.00) [shap -0.231]
- revenue in the prior 12 months (637.00) [shap -0.190]
- largest invoice amount (49.00) [shap -0.094]

### hC62LUJeIccWJJzq

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 1745.22 · Interval: `annual` (conf=0.5693)
- Tier: High risk · Confidence: High (|p-0.5| margin 1.00)

**Main reason:** revenue in the last 12 months (increasing risk)

Risk ↑:
- revenue in the last 12 months (1,392.00) [shap +1.886]
- total invoice lines (21) [shap +1.431]
- billing regularity (4.15) [shap +0.901]
- days since last invoice (362) [shap +0.888]

Risk ↓:
- billing-interval confidence (0.57) [shap -0.434]
- customer tenure (392) [shap -0.302]
- billing events in the last 12 months (4) [shap -0.102]
- invoice-amount variability (0.74) [shap -0.091]

### ryrt5vXAX0NM2018

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 179.76 · Interval: `monthly` (conf=0.5841)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** billing regularity (increasing risk)

Risk ↑:
- billing regularity (3.05) [shap +1.024]
- lifetime revenue (4,176.00) [shap +1.004]
- total invoice lines (12) [shap +0.862]
- days since last invoice (602) [shap +0.689]

Risk ↓:
- billing-interval confidence (0.58) [shap -0.288]
- largest invoice amount (348.00) [shap -0.281]
- revenue in the last 12 months (0.00) [shap -0.223]
- most recent invoice amount (348.00) [shap -0.189]

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
