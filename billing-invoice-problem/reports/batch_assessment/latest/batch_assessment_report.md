# Batch Invoice Assessment Report

_Generated: 2026-07-12T17:17:43.379892+00:00_
_Run dir: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\reports\batch_assessment\run_20260712_171743`_

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
- Recurring-only (>=2 billing days): **False**
- Truncated: True (max_customers=500, allow_full=False)
- Model version: `20260712_032913`
- Mean P(churn): 0.9126
- Total expected $ churn: 91,265.2
- Mean interval confidence: 0.5272

### Recommendation mix (config thresholds)

| Recommendation | Count | Share |
|---|---:|---:|
| Continue | 7 | 1.4% |
| Review | 249 | 49.8% |
| No-Go | 244 | 48.8% |

## 1b. Threshold sensitivity

Same scored probabilities/confidences, re-bucketed under alternate decision thresholds. Rows marked `*` use the config defaults (Continue≤0.45, No-Go≥0.7, min_conf=0.5).

| Cont≤ | No-Go≥ | Min conf | Continue % | Review % | No-Go % | * |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.20 | 0.60 | 0.30 | 0.0 | 28.8 | 71.2 |  |
| 0.20 | 0.70 | 0.30 | 0.0 | 32.6 | 67.4 |  |
| 0.20 | 0.80 | 0.30 | 0.0 | 34.2 | 65.8 |  |
| 0.30 | 0.60 | 0.30 | 0.0 | 28.8 | 71.2 |  |
| 0.30 | 0.70 | 0.30 | 0.0 | 32.6 | 67.4 |  |
| 0.30 | 0.80 | 0.30 | 0.0 | 34.2 | 65.8 |  |
| 0.40 | 0.60 | 0.30 | 1.4 | 27.4 | 71.2 |  |
| 0.40 | 0.70 | 0.30 | 1.4 | 31.2 | 67.4 |  |
| 0.40 | 0.80 | 0.30 | 1.4 | 32.8 | 65.8 |  |
| 0.50 | 0.60 | 0.30 | 4.4 | 24.4 | 71.2 |  |
| 0.50 | 0.70 | 0.30 | 4.4 | 28.2 | 67.4 |  |
| 0.50 | 0.80 | 0.30 | 4.4 | 29.8 | 65.8 |  |
| 0.20 | 0.60 | 0.50 | 0.0 | 48.4 | 51.6 |  |
| 0.20 | 0.70 | 0.50 | 0.0 | 51.2 | 48.8 |  |
| 0.20 | 0.80 | 0.50 | 0.0 | 52.4 | 47.6 |  |
| 0.30 | 0.60 | 0.50 | 0.0 | 48.4 | 51.6 |  |
| 0.30 | 0.70 | 0.50 | 0.0 | 51.2 | 48.8 |  |
| 0.30 | 0.80 | 0.50 | 0.0 | 52.4 | 47.6 |  |
| 0.40 | 0.60 | 0.50 | 1.4 | 47.0 | 51.6 |  |
| 0.40 | 0.70 | 0.50 | 1.4 | 49.8 | 48.8 |  |
| 0.40 | 0.80 | 0.50 | 1.4 | 51.0 | 47.6 |  |
| 0.50 | 0.60 | 0.50 | 4.2 | 44.2 | 51.6 |  |
| 0.50 | 0.70 | 0.50 | 4.2 | 47.0 | 48.8 |  |
| 0.50 | 0.80 | 0.50 | 4.2 | 48.2 | 47.6 |  |

Full sweep (all min_conf × continue × nogo) is in `threshold_sensitivity.json` in this run folder.


## 2. Explainable briefings (top risk)

### 5WMfBPL6Riq9lJeb

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 30.02 · Interval: `monthly` (conf=0.6131)
- Tier: None · Confidence: None

### lqPSqKe82ruz6JSN

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 30.51 · Interval: `monthly` (conf=0.6862)
- Tier: None · Confidence: None

### k5x3RKmrDEcrCQuy

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.49 · Interval: `monthly` (conf=0.5724)
- Tier: None · Confidence: None

### VwwjsRpBjzpemQkD

- Recommendation: **Review** (low_interval_confidence)
- P(churn): 1.0 · Expected $: 3591.93 · Interval: `mixed` (conf=0.2913)
- Tier: None · Confidence: None

### QwIPFHjXUlH7AzRR

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.37 · Interval: `monthly` (conf=0.6344)
- Tier: None · Confidence: None

### ZKqukMSyssgcLGsb

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 2063.35 · Interval: `quarterly` (conf=0.6321)
- Tier: None · Confidence: None

### OWSfpNqqKTsfViQ0

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 8.49 · Interval: `monthly` (conf=0.5579)
- Tier: None · Confidence: None

### LRWftCU3gJL0OM1c

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 33.02 · Interval: `monthly` (conf=0.5994)
- Tier: None · Confidence: None

### hC62LUJeIccWJJzq

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 1745.22 · Interval: `annual` (conf=0.5693)
- Tier: None · Confidence: None

### AsjiD97K8QyIPNXc

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.92 · Interval: `monthly` (conf=0.587)
- Tier: None · Confidence: None

### hXGn4BrCZyrI3pxB

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 2.49 · Interval: `monthly` (conf=0.5724)
- Tier: None · Confidence: None

### n2ne0u6oXlVLAkm5

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 75.55 · Interval: `monthly` (conf=0.6012)
- Tier: None · Confidence: None

### xlUqhCrP5CrlTzn0

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 4.92 · Interval: `monthly` (conf=0.587)
- Tier: None · Confidence: None

### qgrDmgNvDtURUPDD

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 0.9948 · Expected $: 10.44 · Interval: `monthly` (conf=0.6293)
- Tier: None · Confidence: None

### 7IMDlQBsCGZvKMdz

- Recommendation: **No-Go** (high_churn_risk)
- P(churn): 1.0 · Expected $: 636.17 · Interval: `monthly` (conf=0.6857)
- Tier: None · Confidence: None
