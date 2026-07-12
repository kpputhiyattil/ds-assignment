# Mini Report — Billing Interval Inference & Dollar-Churn Credit Decision

**Author / project:** Billing-interval → Continue / Review / No-Go  
**Data:** `Invoices_users.parquet` (22.08M invoices, 562.8K customers, 2023-01-09 → 2026-04-05)  
**Stack:** Python, DuckDB/Polars, LightGBM, SHAP, FastAPI + Streamlit  

This note is the short written summary for the assignment: approach & assumptions, what the analysis revealed, and what to improve with more time. Detailed tables live in `reports/assessment/mini_report.md` and `data/artifacts/ablation_report.json`.

---

## 1. Approach and key assumptions

### Problem
Credit underwriting needs each customer’s **billing interval** (monthly / quarterly / semi-annual / annual / one-time / …), but that field is often missing at first integration. The invoice file has **no ground-truth interval**, so the interval must be inferred and then checked for whether it helps predict **dollar churn**.

### Approach (end-to-end)

1. **EDA / ingestion** — Profile missingness, same-day multi-invoice collapse, history depth, and gap shape before modeling.
2. **Interval inference** — Historical-only rule engine: inter-invoice gaps → calendar multiples (≈30 / 91 / 182 / 365 days), tolerant of skipped periods; emit class + **confidence**.
3. **Leakage-safe features & labels** — Features use only events **before** a snapshot. Dollar churn is defined **independently** of the inferred interval:
   - Dollar loss = `max(0, past-12mo revenue − future-12mo revenue)`
   - Binary churn = future/past revenue ratio below a drop threshold (0.5)
4. **Two comparable churn models** (same algorithm, split, hyperparameters; differ only by features):
   - **Model 1 (baseline):** invoice-derived CORE + GAP features only (no inferred interval)
   - **Model 2 (interval-enhanced):** same + inferred interval one-hots + confidence / diagnostics  
   Each model is two-part: `P(churn)` (calibrated LightGBM) × `E(loss | churn)` (LightGBM L1 severity).
5. **Ablation validation** — Temporal holdout (train earlier snapshot, test later). Two tests:
   - A1: CORE vs CORE+INTERVAL  
   - A2: CORE+GAP vs CORE+GAP+INTERVAL (stricter: does the *derived* label beat raw gaps?)
6. **Decision layer** — Map risk → **Continue / Review / No-Go**, with a data-quality guardrail: low interval confidence → **Review** (never auto No-Go on thin history).
7. **Serving** — Packaged dual models + FastAPI + Streamlit (file upload → explainable dual comparison).

### Key assumptions

| Assumption | Rationale |
|---|---|
| Same-day invoices = one billing event (amounts summed) | 17%+ of customer-days have multiple lines; gaps would be distorted otherwise |
| No CRM interval ground truth | Validate interval **indirectly** via dollar-churn lift, not accuracy vs a label |
| Temporal train/test (not random split) | Mimics production; avoids leakage from future invoices |
| High observed churn rates are definitional | Revenue-drop label on a short-history-heavy base (~62% single-invoice); fair for A/B comparison |
| Gap stats ≠ inferred interval | Gaps are raw spacing; interval is a compressed class + confidence |

---

## 2. What the model / analysis revealed

### Data facts that drove design
- **No nulls** in customer / date / amount — imputation unnecessary; the missing field is **billing interval**.
- **Same-day aggregation is mandatory** (collapse ratio ~0.96 after aggregation).
- **~62% single-invoice customers** → many `insufficient_history` labels and low confidence → Review guardrail is essential.
- After aggregation, only ~24% have ≥2 events (a usable gap signal for cadence).

### Interval value (assignment comparison)

| Test | Result (overall holdout, n≈261k) | Interpretation |
|---|---|---|
| **A1** CORE → CORE+INTERVAL | PR-AUC +0.0005*, dollar-recall@10% +0.0034*, MAE improved | Interval **helps** when only customer-health features exist |
| **A2** CORE+GAP → CORE+GAP+INTERVAL | Small ranking lift; **MAE worse** | Derived interval is largely **redundant** once raw gaps exist |

**Verdict:** **SHIP** the inferred interval as a substitute for missing CRM interval — especially when gap features are not already engineered. It is a useful **compression / explainability** layer, not a large independent accuracy jump on top of gaps.

### Packaged dual models (serving holdout)
WITH vs WITHOUT interval (same temporal test): ranking metrics are essentially tied (PR-AUC ≈ 0.9835 vs 0.9839); WITHOUT is slightly better on expected-loss MAE. Production UI scores **both** so reviewers can see agreement/disagreement per customer.

### Explainability (SHAP)
Top drivers are monetary and recency features (`last_amount`, `max_amount`, `recency_days`, `past12_amount`). **`interval_confidence` ranks ~#5** globally — the model uses inference quality, not only class one-hots.

### Decision mix (illustrative 500-customer batch)
With Continue ≤ 0.45, No-Go ≥ 0.70, min confidence 0.50: **Continue 7 / Review 249 / No-Go 244**. Most mass is Review or No-Go because mean P(churn) is high (~0.91) and many histories fail the confidence guardrail. Dual-model recommendation agreement ≈ **98%**.

---

## 3. What I would improve given more time

1. **Business-calibrated thresholds** — Tune Continue / No-Go cutoffs and min confidence on explicit cost of false Continue vs false No-Go (not only a sensitivity grid).
2. **Segmented models** — Separate recurring (≥2 billing days) vs single-event customers; severity MAE regression in Ablation 2 is concentrated where history is richer.
3. **Stronger severity calibration** — Expected-dollar MAE is the weak point when stacking interval on gaps; isotonic / quantile severity or two-stage loss targeting.
4. **Richer interval inference** — Periodogram / FFT cross-check for high-volume accounts; soft multi-label mixed cadences instead of a single dominant class.
5. **Production hardening** — Drift monitoring (PSI) on gap/interval features, scheduled retrain, Dockerized API, and full-population batch scoring with cost/latency SLAs.
6. **Human evaluation** — Analyst review of disagreeing WITH vs WITHOUT cases to confirm the dual comparison is operationally useful.

---

## 4. How to reproduce / deliverables

```bash
make eda events interval features labels train ablation explain assess package batch-assess
# or: make all
make serve   # FastAPI :8000
make ui      # Streamlit
```

| Deliverable | Path |
|---|---|
| This mini report | `reports/assessment/SUBMISSION_MINI_REPORT.md` |
| Full artifact narrative | `reports/assessment/mini_report.md` |
| Ablation numbers | `data/artifacts/ablation_report.json` |
| Dual serving package | `data/artifacts/serving/churn_decision_model.joblib` |
| Batch triage report | `reports/batch_assessment/latest/` |

**Bottom line:** Infer billing interval from invoices with confidence; train comparable baseline vs interval-enhanced dollar-churn models; ship the interval as a **small, significant** substitute for missing CRM cadence, with Review when confidence is low; serve dual explainable decisions for credit triage.
