# Billing-Interval Inference → Dollar-Churn Prediction

Infer each customer's **billing interval** (monthly / quarterly / semi-annual / annual / one-time / …)
directly from raw invoice history, then measure whether that inferred interval adds **incremental
predictive value** to a **dollar-churn** model — enough to unblock an immediate **Continue / Review /
No-Go** credit decision when the real interval is missing at integration time.

Full design rationale: [`Solution_Blueprint_Billing_Interval.md`](./Solution_Blueprint_Billing_Interval.md).

## Problem framing

| | |
|---|---|
| **ML task** | (1) Unsupervised rule-based interval inference (no ground truth). (2) Two-part dollar-churn model: `P(churn)` × `E(loss \| churn)`. |
| **Target** | Dollar churn = `max(0, past-12mo revenue − future-12mo revenue)`, defined **independently** of the inferred interval to keep validation non-circular. |
| **Validation** | Temporal (train earlier snapshots, test later) + two ablations (core±interval, raw-gap±interval), paired bootstrap. |
| **Data** | `Invoices_users.parquet` — 22.08M invoices, 562.8K customers, 2023-01-09 → 2026-04-05. |

## Tech stack

Python 3.12 · DuckDB (out-of-core aggregation) · Polars/pandas · LightGBM (frequency + severity) ·
scikit-learn · SHAP · pydantic-settings (config) · pytest. FastAPI/Docker serving is an optional extension.

## Project layout

```
configs/training_config.yaml   # single source of truth (paths, seeds, hyperparams)
src/
  config.py                    # typed config loader (pydantic-settings)
  data/                        # ingestion + validation            (Step 2)
  features/                    # interval inference, features, labels (Steps 3-4)
  models/                      # two-part model, ablation, evaluate (Steps 5-6)
  explainability/              # SHAP                                (Step 7)
  serving/                     # decision layer + optional API      (Step 8)
tests/                         # transform/leakage/label/api tests
data/{raw,processed,artifacts} # raw = input; processed/artifacts = generated (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"      # or: pip install -r requirements.txt
pytest                        # smoke tests should pass on a fresh clone
```

## Pipeline run order

Stages are built incrementally. Once implemented, the full run is:

```bash
make events     # raw invoices -> billing_events (same-day aggregation, id check)
make interval   # historical-only interval inference + confidence
make features   # X_core / X_gap / X_interval (leakage-safe, per snapshot)
make labels     # dollar-churn labels (12mo past vs future)
make train      # LightGBM frequency + severity, temporal CV
make ablation   # two ablations x temporal folds + significance verdict
make explain    # SHAP global/local
# or: make all
```

## Build progress

- [x] **Step 1 — Initial setup**: scaffold, config system, dependencies, smoke tests
- [ ] Step 2 — Data loader + validation
- [ ] Step 3 — Interval inference engine
- [ ] Step 4 — Features + dollar-churn labels
- [ ] Step 5 — Two-part model training
- [ ] Step 6 — Ablation harness + business metrics
- [ ] Step 7 — SHAP explainability
- [ ] Step 8 — Decision layer + optional serving

## Reproducibility

All randomness is seeded via `configs/training_config.yaml` (`random_seed`). Config values can be
overridden by environment variables using nested delimiter `__`, e.g. `CHURN__DROP_THRESHOLD=0.4`.
