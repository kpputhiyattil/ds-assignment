# SHAP Explainability Report

_Generated at: 2026-07-11T12:18:56.586086+00:00_

- Model: **`catboost`**
- Landlords in matrix: **14838**
- SHAP sample size: **2000**

> SHAP explains how the model produced its prediction. It does **not** prove that a feature caused company or landlord success.

## Global feature importance (mean |SHAP|)

| Feature | mean |SHAP| |
| --- | ---: |
| `PortfolioMeanClients` | 0.01310 |
| `PortfolioMeanBudget` | 0.00798 |
| `PortfolioActiveRate` | 0.00729 |
| `PortfolioMeanSales` | 0.00622 |
| `PortfolioMeanRetention` | 0.00605 |
| `PortfolioMeanSuccessScore` | 0.00385 |
| `PortfolioSize` | 0.00200 |
| `ActiveCompanies` | 0.00196 |
| `PortfolioStdBudget` | 0.00127 |
| `PortfolioStdSuccessScore` | 0.00116 |
| `OriginCountry` | 0.00067 |
| `OriginCity` | 0.00065 |
| `LandlordAge` | 0.00060 |
| `PreferredIndustry` | 0.00058 |
| `AreaPerCompany` | 0.00022 |
| `PopulationPerCompany` | 0.00020 |
| `Area` | 0.00011 |
| `TotalPopulationAround` | 0.00010 |
| `PopulationDensity` | 0.00007 |

![Global importance](figures/shap/shap_global_importance.png)

## Beeswarm (direction of effects)

![Beeswarm](figures/shap/shap_beeswarm.png)

## Dependence plots

### PortfolioMeanClients

![PortfolioMeanClients](figures/shap/shap_dependence_PortfolioMeanClients.png)

### PortfolioMeanBudget

![PortfolioMeanBudget](figures/shap/shap_dependence_PortfolioMeanBudget.png)

### PortfolioActiveRate

![PortfolioActiveRate](figures/shap/shap_dependence_PortfolioActiveRate.png)

## Case studies

### 1. [GOOD] `LANDLORD_5976`

- AdjustedScore: **0.5467** · OOF pred: **0.4426** · Tenants: **23**

Landlord `LANDLORD_5976` is a **good** case with AdjustedScore=0.5467, model prediction 0.4426 (based on 23 matched tenants). The model associated higher score with: PortfolioMeanClients (+0.0275), PortfolioMeanSales (+0.0127), ActiveCompanies (+0.0076). Downward contributions came from: PortfolioMeanRetention (-0.0134), PortfolioMeanBudget (-0.0064). These are model associations, not proven causal effects.

**Top positive drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanClients` | 0.0275 | 10.86 |
| `PortfolioMeanSales` | 0.0127 | 0.3793 |
| `ActiveCompanies` | 0.0076 | 29 |
| `PortfolioSize` | 0.0074 | 29 |
| `PortfolioStdBudget` | 0.0073 | 1.385e+04 |

**Top negative drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanRetention` | -0.0134 | 0.715 |
| `PortfolioMeanBudget` | -0.0064 | 6229 |

![Waterfall LANDLORD_5976](figures/shap/case_studies/waterfall_good_LANDLORD_5976.png)

### 2. [GOOD] `LANDLORD_8695`

- AdjustedScore: **0.5369** · OOF pred: **0.4450** · Tenants: **25**

Landlord `LANDLORD_8695` is a **good** case with AdjustedScore=0.5369, model prediction 0.4450 (based on 25 matched tenants). The model associated higher score with: PortfolioMeanClients (+0.0246), PortfolioMeanSales (+0.0115), PortfolioSize (+0.0097). Downward contributions came from: PortfolioMeanRetention (-0.0138). These are model associations, not proven causal effects.

**Top positive drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanClients` | 0.0246 | 16.31 |
| `PortfolioMeanSales` | 0.0115 | 0.2564 |
| `PortfolioSize` | 0.0097 | 39 |
| `ActiveCompanies` | 0.0091 | 39 |
| `PortfolioActiveRate` | 0.0045 | 1 |

**Top negative drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanRetention` | -0.0138 | 0.7242 |

![Waterfall LANDLORD_8695](figures/shap/case_studies/waterfall_good_LANDLORD_8695.png)

### 3. [BAD] `LANDLORD_5634`

- AdjustedScore: **0.2484** · OOF pred: **0.2970** · Tenants: **25**

Landlord `LANDLORD_5634` is a **bad** case with AdjustedScore=0.2484, model prediction 0.2970 (based on 25 matched tenants). The model associated higher score with: TotalPopulationAround (+0.0003), LandlordAge (+0.0002). Downward contributions came from: PortfolioMeanBudget (-0.0382), PortfolioMeanSales (-0.0282), PortfolioMeanClients (-0.0218). These are model associations, not proven causal effects.

**Top positive drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `TotalPopulationAround` | 0.0003 | 8.417e+05 |
| `LandlordAge` | 0.0002 | 8 |

**Top negative drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanBudget` | -0.0382 | 1.814e+06 |
| `PortfolioMeanSales` | -0.0282 | 20.04 |
| `PortfolioMeanClients` | -0.0218 | 173.5 |
| `PortfolioActiveRate` | -0.0171 | 0.96 |
| `PortfolioStdBudget` | -0.0155 | 6.213e+06 |

![Waterfall LANDLORD_5634](figures/shap/case_studies/waterfall_bad_LANDLORD_5634.png)

### 4. [BAD] `LANDLORD_10894`

- AdjustedScore: **0.2224** · OOF pred: **0.3546** · Tenants: **6**

Landlord `LANDLORD_10894` is a **bad** case with AdjustedScore=0.2224, model prediction 0.3546 (based on 6 matched tenants). The model associated higher score with: PortfolioMeanClients (+0.0114), PortfolioMeanRetention (+0.0022). Downward contributions came from: PortfolioActiveRate (-0.1288), LandlordAge (-0.0186), PortfolioMeanSales (-0.0186). These are model associations, not proven causal effects.

**Top positive drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanClients` | 0.0114 | 24.83 |
| `PortfolioMeanRetention` | 0.0022 | 0.228 |

**Top negative drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioActiveRate` | -0.1288 | 0.3333 |
| `LandlordAge` | -0.0186 | 22 |
| `PortfolioMeanSales` | -0.0186 | 21.5 |
| `PortfolioMeanBudget` | -0.0079 | 4421 |
| `ActiveCompanies` | -0.0051 | 11 |

![Waterfall LANDLORD_10894](figures/shap/case_studies/waterfall_bad_LANDLORD_10894.png)

### 5. [NEUTRAL] `LANDLORD_3491`

- AdjustedScore: **0.4382** · OOF pred: **0.3883** · Tenants: **5**

Landlord `LANDLORD_3491` is a **neutral** case with AdjustedScore=0.4382, model prediction 0.3883 (based on 5 matched tenants). The model associated higher score with: PortfolioMeanSales (+0.0049), PortfolioActiveRate (+0.0039), PortfolioMeanBudget (+0.0029). Downward contributions came from: PortfolioMeanSuccessScore (-0.0022), ActiveCompanies (-0.0012), PortfolioSize (-0.0007). These are model associations, not proven causal effects.

**Top positive drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanSales` | 0.0049 | 1.833 |
| `PortfolioActiveRate` | 0.0039 | 1 |
| `PortfolioMeanBudget` | 0.0029 | 760.4 |
| `PreferredIndustry` | 0.0012 | Consumer Durables |
| `PortfolioMeanRetention` | 0.0008 | 0.4191 |

**Top negative drivers**

| Feature | SHAP | Value |
| --- | ---: | --- |
| `PortfolioMeanSuccessScore` | -0.0022 | 0.5816 |
| `ActiveCompanies` | -0.0012 | 6 |
| `PortfolioSize` | -0.0007 | 6 |
| `TotalPopulationAround` | -0.0001 | 1.674e+05 |

![Waterfall LANDLORD_3491](figures/shap/case_studies/waterfall_neutral_LANDLORD_3491.png)

---
_Use wording such as “associated with” or “contributed to the prediction”; avoid causal claims unless the study design supports them._
