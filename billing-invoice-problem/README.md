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
  data/                        # ingestion + validation + EDA         (Step 2)
  features/                    # interval inference, features, labels (Steps 3-4)
  models/                      # two-part model, ablation, evaluate (Steps 5-6)
  explainability/              # SHAP                                (Step 7)
  serving/                     # decision layer + optional API      (Step 8)
  reporting/                   # assessment mini-report from artifacts
  scripts/                     # run_eda / run_assessment CLIs
tests/                         # transform/leakage/label/api tests
reports/{eda,assessment}/      # generated EDA + mini-report (committed narrative)
data/{raw,processed,artifacts} # raw = input; processed/artifacts = generated (gitignored)
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -e ".[dev,serving]"   # ML + FastAPI + Streamlit
# or: pip install -r requirements.txt  &&  pip install -e ".[serving]"
pytest
```

Requires **Python ≥ 3.11**. Always use the **venv** interpreter for uvicorn/streamlit
(not a system-wide `python.exe`), so dependencies resolve correctly.

## Sharing this project with someone else

### What to send

| Item | Required? | Notes |
|---|---|---|
| Project code (repo / zip) | Yes | Source, configs, tests, reports |
| `data/raw/Invoices_users.parquet` | Yes | **Gitignored** — must share separately |
| `data/processed/` + `data/artifacts/` | Optional | Skip if you want them to retrain; include to skip the long pipeline |

### Full rebuild (from raw invoices)

Put the parquet at `data/raw/Invoices_users.parquet`, activate the venv, then either:

```bash
make all
```

or run each stage (Windows-friendly; same order as `make all`):

```bash
# 1) EDA BEFORE training
python -m src.scripts.run_eda

# 2) Feature pipeline
python -m src.data.ingestion
python -m src.features.interval
python -m src.features.build
python -m src.features.labels

# 3) Train + validate interval value
python -m src.models.train
python -m src.models.ablation
python -m src.explainability.shap_utils

# 4) Assessment AFTER ablation/explain (needs those artifacts)
python -m src.scripts.run_assessment

# 5) Package dual models + batch report
python -m src.serving.package
python -m src.scripts.run_batch_assess
```

**Order note:** `run_eda` belongs **before** training. `run_assessment` belongs
**after** `ablation` + `explain` — it reads those artifacts for the with-vs-without
interval comparison and cannot produce that earlier.

### Serve the API + UI (2 terminals)

```bash
# Terminal 1 — API (venv python)
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py
```

Or: `make serve` and `make ui`.

### Faster path (artifacts already included)

If you also sent `data/artifacts/serving/churn_decision_model.joblib`:

```bash
pip install -e ".[dev,serving]"
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
streamlit run app/streamlit_app.py
```

They still need invoice data to upload (sample CSV/Parquet or the full raw file).

### Sanity checks

| Check | Expected |
|---|---|
| `GET http://127.0.0.1:8000/health` | `"status": "ok"` |
| `GET http://127.0.0.1:8000/model/info` | `"dual_models": true` |
| EDA | `reports/eda/` |
| Assessment mini-report | `reports/assessment/mini_report.md` |
| Batch assessment | `reports/batch_assessment/latest/` |

## Pipeline run order

Stages are built incrementally. Once implemented, the full run is:

```bash
make eda        # FIRST — invoice EDA (missingness, same-day, gaps) -> reports/eda/
make events     # raw invoices -> billing_events (same-day aggregation, id check)
make interval   # historical-only interval inference + confidence
make features   # X_core / X_gap / X_interval (leakage-safe, per snapshot)
make labels     # dollar-churn labels (12mo past vs future)
make train      # LightGBM frequency + severity, temporal CV
make ablation   # two ablations x temporal folds + significance verdict
make explain    # SHAP global/local
make assess     # AFTER train/ablation — mini-report from artifacts
make package    # dual with/without-interval serving artifact
make batch-assess
# or: make all
```

**Order note:** `make eda` belongs **before** training (design / data understanding).
`make assess` belongs **after** `ablation` + `explain` — it reads those artifacts and
cannot produce the with-vs-without comparison earlier.

## Build progress

- [x] **Step 1** — Initial setup: scaffold, config system, dependencies, smoke tests
- [x] **Step 2** — Data loader + validation (same-day aggregation -> billing_events)
- [x] **Step 3** — Interval inference engine (8-class, skipped-period matching, confidence)
- [x] **Step 4** — Leakage-safe features (core/gap/interval) + dollar-churn labels
- [x] **Step 5** — Two-part model: calibrated frequency + severity, temporal eval
- [x] **Step 6** — Ablation harness (bootstrap + DeLong) -> interval-value verdict
- [x] **Step 7** — TreeSHAP global importance + per-customer decision briefings
- [x] **Step 8** — Decision layer + packaged artifact + FastAPI + drift monitoring

### Headline result

The inferred interval delivers a small but **statistically significant** lift over
customer-health features (Ablation 1) -> it is "good enough" to substitute for the
missing real interval and unblock the Continue/Review/No-Go decision. It is largely
**redundant with raw gap features** (Ablation 2), so it is optional if those are
already engineered. See `data/artifacts/ablation_report.json` and the written
comparison narrative in `reports/assessment/mini_report.md` (`make assess`).
That file is the **only** mini-report to send (approach, with-vs-without summary,
ablation detail, improvements) — regenerated by `make assess`.

## Serving

Package the feature-engineering path + calibrated models + SHAP capability into
one artifact, then serve decisions via FastAPI. Streamlit is the analyst UI and
calls the API (manual entry or CSV/Parquet upload).

```bash
pip install -e ".[serving]"
make package                       # -> data/artifacts/serving/churn_decision_model.joblib
make batch-assess                  # score full Invoices_users.parquet + report
make serve                         # FastAPI on :8000
# in another terminal:
make ui                            # Streamlit review app
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + model version |
| `GET /model/info` | Package checklist / feature list |
| `POST /assess` | One customer (JSON invoices) → explainable Continue/Review/No-Go |
| `POST /assess/file` | Upload CSV/Parquet → **capped** batch scores + top-N SHAP briefings |

Interactive uploads are guarded (config `serving.*`):

- default **500** customers (hard max **5000**)
- max upload **128 MB** (covers `Invoices_users.parquet` ~67 MB)
- lazy parquet/CSV scan + filter **before** collecting rows

Full-file offline scoring (opt-in only):

```bash
make batch-assess          # capped (safe default)
make batch-assess-full     # entire file — memory-intensive
```

```bash
# Single customer
curl -X POST http://localhost:8000/assess -H "Content-Type: application/json" -d "{\"customer_id\":\"c1\",\"snapshot\":\"2025-03-05\",\"invoices\":[{\"date\":\"2024-01-15\",\"amount\":120},{\"date\":\"2024-02-15\",\"amount\":120}]}"

# Full file (from Streamlit, or curl multipart)
# POST /assess/file  file=@Invoices_users.parquet  snapshot=2025-03-05  explain_top_n=10
```

Batch report output (timestamped; previous runs kept):

```text
reports/batch_assessment/run_YYYYMMDD_HHMMSS/
reports/batch_assessment/latest/          # pointer to newest run
```

Each batch report also embeds the **interval-value ablation summary** from
`data/artifacts/ablation_report.json` (assignment part 3). That comparison is
produced by `make ablation`, not by re-scoring the batch file.

The decision layer maps churn risk → Continue / Review / No-Go, with a
**data-quality guardrail**: interval_confidence below
`decision.min_interval_confidence_for_auto` routes to Review (never auto No-Go).
Feature drift is monitored via PSI (`src/serving/monitoring.py`).

## Reproducibility

All randomness is seeded via `configs/training_config.yaml` (`random_seed`). Config values can be
overridden by environment variables using nested delimiter `__`, e.g. `CHURN__DROP_THRESHOLD=0.4`.
