# Good Landlord / Bad Landlord — Landlord Quality Scoring

A production-grade ML pipeline that estimates whether a landlord is associated with better business outcomes for tenant companies. Rather than labelling a landlord "good" because many tenants are currently active, the system:

1. **Estimates each company's expected performance** given its own characteristics (industry, age, budget, etc.)
2. **Computes residuals** (actual minus expected) to isolate the landlord-specific contribution
3. **Aggregates residuals per landlord** with shrinkage for small tenant counts
4. **Trains a CatBoost regression model** to predict the adjusted landlord quality score from landlord features alone — enabling scoring of new landlords

---

## Repository Structure

```
good-landlord-model/
├── configs/
│   └── training_config.yaml       # all hyperparams, paths, seeds
├── src/
│   ├── data/
│   │   ├── ingestion.py           # load raw parquets, build bridge table
│   │   └── validation.py          # schema + data-quality checks
│   ├── features/
│   │   └── transforms.py          # shared train/serve feature transforms
│   ├── targets/
│   │   └── construction.py        # company success target design
│   ├── models/
│   │   ├── company_baseline.py    # OOF company model + residuals
│   │   ├── train.py               # landlord-level CatBoost training
│   │   └── evaluate.py            # metrics, lift, segment analysis
│   └── explainability/
│       └── shap_utils.py          # global + local SHAP outputs
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_target_construction.ipynb
│   ├── 03_company_baseline_model.ipynb
│   ├── 04_landlord_modeling.ipynb
│   └── 05_explainability_report.ipynb
├── data/
│   ├── raw/          # place LandLords.parquet and Companies.parquet here (gitignored)
│   ├── processed/    # intermediate outputs (gitignored)
│   └── artifacts/    # saved model artifacts (gitignored)
├── reports/
│   ├── figures/      # SHAP plots and evaluation charts
│   └── mini_report.md
├── tests/
│   ├── test_ingestion.py
│   ├── test_validation.py
│   ├── test_features.py
│   └── test_targets.py
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

Each step below corresponds to one git commit. Run them in order:

```bash
# Validate raw data and inspect quality report
python -m src.data.validation

# Build company success targets
python -m src.targets.construction

# Fit OOF company baseline + compute adjusted landlord scores
python -m src.models.company_baseline

# Train landlord CatBoost model
python -m src.models.train

# Evaluate + generate SHAP explainability outputs
python -m src.models.evaluate
python -m src.explainability.shap_utils
```

Or explore interactively via the numbered notebooks in `notebooks/`.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Analytical Approach

| Stage | What it does |
|-------|-------------|
| Target construction | Maps `CompanyStatus` to binary success; falls back to a composite score (sales efficiency, retention, conversion, momentum) normalized within industry |
| Company baseline | Logistic Regression trained on company-only features; generates OOF predicted probabilities to avoid data leakage |
| Adjusted landlord score | `residual = actual − expected`; aggregate per landlord with Empirical-Bayes shrinkage `N / (N + m)` toward global mean |
| Landlord model | CatBoostRegressor on landlord features (age, density, area-per-company, population-per-company, industry alignment); GroupKFold by `LandLordID` |
| Output | `PredictedQualityScore` (0–100), `PercentileRank`, `QualityBand`, `TenantCount`, `ConfidenceLevel`, SHAP drivers |

### Key leakage controls

- All companies for the same landlord stay in the same CV fold (`GroupKFold(group=LandLordID)`)
- Company residuals are generated **out-of-fold** before aggregation
- Target encodings computed within folds only

---

## Final Output Schema

| Field | Description |
|-------|-------------|
| `LandLordID` | Unique landlord identifier |
| `PredictedQualityScore` | Model output rescaled 0–100 |
| `PercentileRank` | Position among all scored landlords |
| `QualityBand` | `good` / `neutral` / `bad` |
| `ProbabilityGood` | Calibrated probability (classification head) |
| `TenantCount` | Matched tenant companies used in historical evaluation |
| `ConfidenceLevel` | `high` / `medium` / `low` based on sample size |
| `TopPositiveDrivers` | Top SHAP features increasing score |
| `TopNegativeDrivers` | Top SHAP features decreasing score |
| `ModelVersion` | Git commit + timestamp |

---

## Limitations

- Model estimates **association**, not causation — landlord quality is confounded by tenant selection, location economics, and unobserved rental terms
- Survivorship bias: failed companies may be absent from `AllCompanyID`
- Single snapshot data; no pre/post tenancy comparison is possible
- Landlords with fewer than ~5 tenants receive high shrinkage; treat their scores with caution

---

## Assumptions to Confirm Before Running

1. What exact values appear in `CompanyStatus`?
2. Is `AllCompanyID` a Python list, comma-separated string, or other format?
3. Can a company appear under multiple landlords?
4. Are there multiple historical snapshots per entity, or one row per ID?
5. Does higher `Rank` mean better or worse customer feedback?
6. Are `origin` fields permitted for modeling or for analysis/fairness-audit only?
