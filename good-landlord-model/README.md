# Good Landlord / Bad Landlord — Landlord Quality Scoring

A production-grade ML pipeline that estimates whether a landlord is associated with better business outcomes for tenant companies. Rather than labelling a landlord "good" because many tenants are currently active, the system:

1. **Estimates each company's expected performance** given its own characteristics (industry, age, budget, etc.)
2. **Computes residuals** (actual minus expected) to isolate the landlord-specific contribution
3. **Aggregates residuals per landlord** with shrinkage for small tenant counts
4. **Trains landlord regression models** (CatBoost recommended) to predict the adjusted quality score — enabling scoring of new landlords
5. **Explains predictions** with SHAP global and case-study outputs

---

## Repository Structure

```
good-landlord-model/
├── configs/
│   └── training_config.yaml       # all hyperparams, paths, seeds
├── scripts/                       # pipeline entrypoints
│   ├── run_eda.py
│   ├── run_company_targets.py
│   ├── run_landlord_models.py
│   ├── compare_landlord_models.py
│   ├── run_shap_explainability.py
│   ├── export_landlord_scorer.py
│   └── write_mini_report.py
├── service/                       # FastAPI + UI scoring desk
│   ├── app/main.py
│   ├── templates/
│   ├── static/
│   └── README.md
├── src/
│   ├── config.py
│   ├── utils/reproducibility.py   # set_seed, repo_relpath
│   ├── data/                      # ingestion, validation, eda
│   ├── features/transforms.py
│   ├── targets/construction.py
│   ├── models/                    # baseline, train, evaluate, scorer
│   └── explainability/shap_utils.py
├── data/
│   ├── raw/          # LandLords.parquet, Companies.parquet (gitignored)
│   ├── processed/    # intermediate outputs (gitignored)
│   └── artifacts/    # saved models (gitignored)
├── reports/
│   ├── figures/
│   ├── mini_report.md
│   ├── model_comparison_report.md
│   └── shap_explainability_report.md
├── tests/
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

### 2. Place raw data

```
data/raw/LandLords.parquet
data/raw/Companies.parquet
```

### 3. Run the full pipeline

```bash
# Validate raw data
python -m src.data.validation

# Pipeline runners (scripts/ = entrypoints; src/ = library)
python scripts/run_eda.py
python scripts/run_company_targets.py
python scripts/run_landlord_models.py
python scripts/compare_landlord_models.py
python scripts/run_shap_explainability.py
python scripts/export_landlord_scorer.py
python scripts/write_mini_report.py

# Optional: scoring UI (FastAPI)
pip install -r service/requirements.txt
uvicorn service.app.main:app --reload --port 8080
# open http://127.0.0.1:8080
```

All runners call `set_seed(cfg["seed"])` (default **42**) so NumPy / Python RNGs align with `configs/training_config.yaml`.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Analytical Approach

| Stage | What it does |
|-------|-------------|
| Target construction | Maps `CompanyStatus` to binary success; composite score normalized within industry |
| Company baseline | Logistic Regression on company-only features; OOF probs (GroupKFold by landlord) |
| Adjusted landlord score | Residual aggregation + Empirical-Bayes shrinkage `N / (N + m)` |
| Landlord model | CatBoost / XGBoost / LightGBM / RF; GroupKFold by `LandLordID` |
| Explainability | TreeSHAP global importance, dependence plots, good/bad/neutral case studies |
| Serveable artifact | `LandlordScorer` packs **model + preprocessing + feature engineering + SHAP** |
| Output | Adjusted / predicted scores, bands, SHAP drivers, mini report |

### Serving the combined artifact

```python
from src.models.scorer import LandlordScorer

scorer = LandlordScorer.load("data/artifacts/landlord_scorer.joblib")
print(scorer.package_manifest())  # model / preprocessing / FE / shap

# End-to-end: feature engineering -> preprocess -> predict -> SHAP
rows, pipeline_info = scorer.score_end_to_end(upload_df, top_k=5)

# Or score a ready feature matrix directly
result = scorer.score_with_explanation(landlord_feature_rows, top_k=5)
# each row: PredictedScore, QualityBand, ConfidenceLevel,
#           TopPositiveDrivers, TopNegativeDrivers, ModelVersion
```

### Key leakage controls

- All companies for the same landlord stay in the same CV fold (`GroupKFold(group=LandLordID)`)
- Company residuals are generated **out-of-fold** before aggregation
- Target encodings computed within folds only

---

## Reproducibility

| Knob | Location |
|------|----------|
| Global seed | `configs/training_config.yaml` → `seed: 42` |
| Model RNGs | `landlord_model.*.random_seed` / `random_state` |
| Runtime seeding | `src.utils.reproducibility.set_seed` (called by every `scripts/` runner) |
| Paths in JSON | Stored relative to repo root via `repo_relpath` |

Docker:

```bash
docker build -t good-landlord-model .
docker run --rm good-landlord-model pytest tests/ -q
```

---

## Final Output Schema

| Field | Description |
|-------|-------------|
| `LandLordID` | Unique landlord identifier |
| `PredictedQualityScore` / `AdjustedScore` | Model / EB-shrunk quality signal |
| `PercentileRank` | Position among scored landlords |
| `QualityBand` | `good` / `neutral` / `bad` |
| `TenantCount` | Matched tenant companies |
| `ConfidenceLevel` | `high` / `medium` / `low` by sample size |
| `TopPositiveDrivers` / `TopNegativeDrivers` | Local SHAP features |
| `ModelVersion` | Best model type + config seed |

---

## Limitations

- Model estimates **association**, not causation
- Survivorship bias: failed companies may be absent from `AllCompanyID`
- Single snapshot; no pre/post tenancy comparison
- Landlords with fewer than ~5 tenants receive high shrinkage

---

## Assumptions Confirmed on This Dataset

1. `CompanyStatus` values mapped in `training_config.yaml` (`status_positive` / `status_negative` / `status_exclude`)
2. `AllCompanyID` stored as Python list reprs — handled by `ast.literal_eval` in auto mode
3. Bridge↔Companies match rate ~70%; unmatched high IDs are a data gap
4. Higher composite / residual scores treated as better landlord-associated outcomes
