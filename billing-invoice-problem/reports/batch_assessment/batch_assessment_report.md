# Batch Invoice Assessment Report

_Generated: 2026-07-12T15:59:55.247148+00:00_

## Summary

- Snapshot: **2025-03-05**
- Customers scored: **500**
- Model version: `20260712_032913`
- Mean P(churn): 0.9126
- Total expected $ churn: 91,265.29
- Mean interval confidence: 0.5272

### Recommendation mix

| Recommendation | Count | Share |
|---|---:|---:|
| Continue | 0 | 0.0% |
| Review | 256 | 51.2% |
| No-Go | 244 | 48.8% |

## Explainable briefings (top risk)

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

### gQpEa6o4LoyyP13G

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 29.8 · Interval: `monthly` (conf=0.6857)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (28) [shap +2.867]
- days since last invoice (405) [shap +0.874]
- average invoice amount (49.00) [shap +0.435]
- number of billing events (28) [shap +0.425]

Risk ↓:
- customer tenure (368) [shap -0.620]
- billing-interval confidence (0.69) [shap -0.355]
- largest invoice amount (49.00) [shap -0.145]
- most recent invoice amount (49.00) [shap -0.138]

### xOlth8U4DrsAvvcw

- Recommendation: **Review** (low_interval_confidence)
- P(churn): 0.9948 · Expected $: 1566.55 · Interval: `mixed` (conf=0.2752)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** revenue in the last 12 months (increasing risk)

Risk ↑:
- revenue in the last 12 months (1,048.00) [shap +1.383]
- customer tenure (708) [shap +0.909]
- total invoice lines (14) [shap +0.901]
- revenue in the prior 12 months (2,088.00) [shap +0.730]

Risk ↓:
- days since last invoice (32) [shap -0.888]
- most recent invoice amount (696.00) [shap -0.313]
- largest invoice amount (696.00) [shap -0.220]
- billing-interval confidence (0.28) [shap -0.140]

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

### bvBpq44pFzTNNhnJ

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 8.49 · Interval: `monthly` (conf=0.5595)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (13) [shap +1.751]
- days since last invoice (660) [shap +0.654]
- average invoice amount (49.00) [shap +0.496]
- number of billing events (13) [shap +0.473]

Risk ↓:
- billing-interval confidence (0.56) [shap -0.478]
- most recent invoice amount (49.00) [shap -0.155]
- lifetime revenue (637.00) [shap -0.086]
- largest invoice amount (49.00) [shap -0.078]

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

### k5x3RKmrDEcrCQuy

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

### n2ne0u6oXlVLAkm5

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 75.55 · Interval: `monthly` (conf=0.6012)
- Tier: High risk · Confidence: High (|p-0.5| margin 0.99)

**Main reason:** total invoice lines (increasing risk)

Risk ↑:
- total invoice lines (14) [shap +1.550]
- days since last invoice (572) [shap +0.766]
- revenue trend (recent vs prior year) (0.00) [shap +0.577]
- billing regularity (0.96) [shap +0.570]

Risk ↓:
- billing-interval confidence (0.60) [shap -0.376]
- most recent invoice amount (147.00) [shap -0.262]
- largest invoice amount (147.00) [shap -0.176]
- typical days between invoices (13) [shap -0.130]

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
