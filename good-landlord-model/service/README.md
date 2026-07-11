# Landlord Quality Scoring Service

Standalone FastAPI app + UI that loads the packed **LandlordScorer**
(`data/artifacts/landlord_scorer.joblib`) and returns predictions with
human-readable SHAP explanations.

Uploads are run through the **same feature-engineering path as training** when
they look like raw landlord data (or landlord IDs):

`build_model_base` → company targets → `landlord_features` → `portfolio_features`

using the project’s Companies table for portfolio aggregations.

## Prerequisites

From the **project root** (`good-landlord-model/`):

```bash
pip install -e ".[dev]"
pip install -r service/requirements.txt
python scripts/export_landlord_scorer.py
```

## Run

```bash
uvicorn service.app.main:app --reload --host 0.0.0.0 --port 8080
```

Open **http://127.0.0.1:8080**

## What to upload

| Input | What happens |
|-------|----------------|
| Raw **LandLords** rows (`LandLordID`, `AllCompanyID`, …) | Full FE pipeline, then score |
| **LandLordID** list only | Look up landlords in project data → FE → score |
| Pre-built **feature matrix** | Score directly (skips FE) |

## API

| Method | Path | Body |
|--------|------|------|
| `GET` | `/api/health` | — |
| `GET` | `/api/schema` | — |
| `POST` | `/api/score/json` | `{ "landlords": [ {...} ], "top_k": 5 }` |
| `POST` | `/api/score/file` | `multipart/form-data`: `file`, `top_k` |
| `GET` | `/docs` | OpenAPI UI |

### Example: raw landlords via file

Upload a CSV/Parquet with the same columns as `data/raw/LandLords.parquet`
(or a subset of rows). The service joins Companies + builds portfolio features
exactly as in `scripts/run_landlord_models.py`.

### Example: landlord IDs

```bash
python -c "import pandas as pd; pd.DataFrame({'LandLordID':['LANDLORD_0001','LANDLORD_0002']}).to_csv('ids.csv', index=False)"
curl -X POST http://127.0.0.1:8080/api/score/file -F "file=@ids.csv" -F "top_k=5"
```
