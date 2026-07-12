# Good Landlord / Bad Landlord — Landlord Quality Scoring

Score landlords by how well their tenant companies perform **after adjusting for company
characteristics** — isolating the landlord-specific signal from confounders like industry,
budget, and company age.

## Problem framing

| | |
|---|---|
| **ML task** | (1) Company baseline (logistic regression, OOF) → residuals. (2) Landlord regression (CatBoost / XGB / LGBM / RF) on aggregated + shrunk residuals. |
| **Target** | `AdjustedScore` — Empirical-Bayes–shrunk mean residual per landlord: `N/(N+m) * raw + m/(N+m) * global_mean`. |
| **Validation** | GroupKFold by `LandLordID` everywhere (prevents leakage from shared tenants). |
| **Data** | `LandLords.parquet` + `Companies.parquet` — linked via `AllCompanyID` bridge (~70 % match rate). |

## Tech stack

Python 3.11 · pandas / pyarrow · scikit-learn · CatBoost · XGBoost · LightGBM ·
SHAP · PyYAML · matplotlib / seaborn · FastAPI + Streamlit serving · Docker · pytest.

## Project layout

```
configs/training_config.yaml   # single source of truth (paths, seeds, hyperparams)
scripts/                       # pipeline entrypoints (run in order)
  run_eda.py
  run_company_targets.py
  run_landlord_models.py
  compare_landlord_models.py
  run_shap_explainability.py
  export_landlord_scorer.py
  write_mini_report.py
src/
  config.py                    # typed YAML config loader
  data/                        # ingestion, validation, EDA
  targets/                     # CompanyIsActive + SuccessScore
  features/                    # landlord + portfolio feature engineering
  models/                      # company baseline, train, evaluate, scorer
  explainability/              # TreeSHAP, glossary, narratives
service/                       # FastAPI + HTML scoring desk
app/streamlit_app.py           # Streamlit analyst UI (calls API)
tests/                         # 13 pytest modules (synthetic data, no parquet needed)
data/{raw,processed,artifacts} # raw = input; processed/artifacts = generated (gitignored)
reports/                       # EDA, comparison, SHAP, mini-report (committed narrative)
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -e ".[dev,serving]"   # ML + FastAPI + Streamlit
# or: pip install -r requirements.txt  &&  pip install -r service/requirements.txt
pytest
```

Requires **Python >= 3.10**. Always use the **venv** interpreter for uvicorn/streamlit
(not a system-wide `python.exe`), so dependencies resolve correctly.

## Sharing this project with someone else

### What to send

| Item | Required? | Notes |
|---|---|---|
| Project code (repo / zip) | Yes | Source, configs, tests, reports |
| `data/raw/LandLords.parquet` | Yes | **Gitignored** — must share separately |
| `data/raw/Companies.parquet` | Yes | **Gitignored** — must share separately |
| `data/processed/` + `data/artifacts/` | Optional | Skip if you want them to retrain; include to skip the long pipeline |

### Full rebuild (from raw data)

Place parquet files at `data/raw/`, activate the venv, then:

```bash
# 1) Data validation + EDA (BEFORE training)
python -m src.data.validation
python scripts/run_eda.py

# 2) Target + model pipeline
python scripts/run_company_targets.py
python scripts/run_landlord_models.py

# 3) Compare + explain
python scripts/compare_landlord_models.py
python scripts/run_shap_explainability.py

# 4) Package scorer + report (AFTER compare/explain — reads those artifacts)
python scripts/export_landlord_scorer.py
python scripts/write_mini_report.py
```

**Order note:** `run_eda` belongs **before** training. `export_landlord_scorer` and
`write_mini_report` belong **after** compare + explain — they read those artifacts.

### Serve the API + UI (2 terminals)

```bash
# Terminal 1 — FastAPI
python -m uvicorn service.app.main:app --host 127.0.0.1 --port 8080

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py
```

### Faster path (artifacts already included)

If you also sent `data/artifacts/landlord_scorer.joblib`:

```bash
pip install -e ".[dev,serving]"
python -m uvicorn service.app.main:app --host 127.0.0.1 --port 8080
streamlit run app/streamlit_app.py
```

They still need landlord data to upload (sample CSV/Parquet or the raw files).

### Sanity checks

| Check | Expected |
|---|---|
| `GET http://127.0.0.1:8080/api/health` | `"status": "ok"` |
| `GET http://127.0.0.1:8080/api/schema` | Feature list + model info |
| FastAPI HTML desk | `http://127.0.0.1:8080` |
| Streamlit UI | `http://localhost:8501` |
| EDA report | `reports/eda_report.md` |
| Mini-report | `reports/mini_report.md` |

## Pipeline run order

Stages are built incrementally. Full run:

```bash
python -m src.data.validation           # data quality → data/processed/data_quality_report.json
python scripts/run_eda.py               # EDA → reports/eda_report.{json,md}
python scripts/run_company_targets.py   # targets → data/processed/company_targets.parquet
python scripts/run_landlord_models.py   # baseline + FE + train + eval → models + reports
python scripts/compare_landlord_models.py   # best-model pick → reports/model_comparison_*
python scripts/run_shap_explainability.py   # SHAP → reports/shap_*, figures/shap/
python scripts/export_landlord_scorer.py    # packaged artifact → data/artifacts/landlord_scorer.joblib
python scripts/write_mini_report.py         # executive summary → reports/mini_report.md
```

## Analytical approach

| Stage | What it does |
|---|---|
| Target construction | Maps `CompanyStatus` → binary; composite score normalized within industry |
| Company baseline | Logistic Regression on company-only features; OOF probs (GroupKFold by landlord) |
| Adjusted landlord score | Residual aggregation + Empirical-Bayes shrinkage `N / (N + m)` |
| Landlord model | CatBoost / XGBoost / LightGBM / RF; GroupKFold by `LandLordID` |
| Explainability | TreeSHAP global importance, dependence plots, good/bad/neutral case studies |
| Serveable artifact | `LandlordScorer` packs **model + preprocessing + feature engineering + SHAP** |

### Key leakage controls

- All companies for the same landlord stay in the same CV fold (`GroupKFold(group=LandLordID)`)
- Company residuals are generated **out-of-fold** before aggregation
- Target encodings computed within folds only

## Serving

Package the feature-engineering path + model + SHAP into one artifact, then serve
via FastAPI. Streamlit is the analyst UI and calls the API (manual entry or CSV/Parquet upload).

```bash
pip install -e ".[serving]"
python scripts/export_landlord_scorer.py      # → data/artifacts/landlord_scorer.joblib
python -m uvicorn service.app.main:app --port 8080    # FastAPI on :8080
# in another terminal:
streamlit run app/streamlit_app.py                     # Streamlit on :8501
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness + model version + package manifest |
| `GET /api/schema` | Feature list, glossary, model metadata |
| `POST /api/score/json` | One or more landlords (JSON) → explainable scores |
| `POST /api/score/file` | Upload CSV/Parquet → batch scores + SHAP drivers |
| `GET /` | Built-in HTML scoring desk |

Interactive uploads are capped via env vars:

| Variable | Default | Purpose |
|---|---|---|
| `LANDLORD_SCORER_PATH` | `data/artifacts/landlord_scorer.joblib` | Path to scorer artifact |
| `LANDLORD_SCORE_MAX_ROWS` | `100` | Max rows per request |
| `LANDLORD_SCORE_TOP_K` | `5` | SHAP drivers per landlord |
| `LANDLORD_API_URL` | `http://127.0.0.1:8080` | API base for Streamlit |

### Using the scorer in Python

```python
from src.models.scorer import LandlordScorer

scorer = LandlordScorer.load("data/artifacts/landlord_scorer.joblib")
print(scorer.package_manifest())

rows, pipeline_info = scorer.score_end_to_end(upload_df, top_k=5)
result = scorer.score_with_explanation(landlord_feature_rows, top_k=5)
```

## Final output schema

| Field | Description |
|---|---|
| `LandLordID` | Unique landlord identifier |
| `PredictedQualityScore` / `AdjustedScore` | Model / EB-shrunk quality signal |
| `PercentileRank` | Position among scored landlords |
| `QualityBand` | `good` / `neutral` / `bad` |
| `TenantCount` | Matched tenant companies |
| `ConfidenceLevel` | `high` / `medium` / `low` by sample size |
| `TopPositiveDrivers` / `TopNegativeDrivers` | Local SHAP features |
| `ModelVersion` | Best model type + config seed |

## Reproducibility

All randomness is seeded via `configs/training_config.yaml` (`seed: 42`).
Every `scripts/` runner calls `set_seed(cfg["seed"])` at startup.

Docker:

```bash
docker build -t good-landlord-model .
docker run --rm good-landlord-model pytest tests/ -q
```

## Limitations

- Model estimates **association**, not causation
- Survivorship bias: failed companies may be absent from `AllCompanyID`
- Single snapshot; no pre/post tenancy comparison
- Landlords with fewer than ~5 tenants receive high shrinkage
- Bridge↔Companies match rate ~70 %; unmatched IDs are a data gap
