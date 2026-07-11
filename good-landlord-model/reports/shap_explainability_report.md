# SHAP Explainability Report

_Generated at: 2026-07-11T12:52:53.522873+00:00_

- Model: **`catboost`**
- Landlords in matrix: **14838**
- SHAP sample size: **1500**

> SHAP explains how the model produced its prediction in plain language. It does **not** prove that a feature caused company or landlord success.

## Feature glossary

Technical column names are mapped to human-readable meanings below (also applied in case-study narratives).

| Column | Meaning | What it measures |
| --- | --- | --- |
| `LandlordAge` | Landlord age (years) | How long the landlord has been operating (years since founding). |
| `Area` | Property / campus area | Physical area associated with the landlord. |
| `TotalPopulationAround` | Nearby population | Population living around the landlord’s location. |
| `ActiveCompanies` | Reported active companies | Number of companies the landlord reports as active tenants. |
| `PopulationDensity` | Population density | Nearby population divided by area (crowding / urban intensity). |
| `AreaPerCompany` | Area per company | Space available per reported company (area ÷ active companies). |
| `PopulationPerCompany` | Population per company | Nearby population per reported company. |
| `PreferredIndustry` | Preferred industry | Industry focus the landlord prefers for tenants. |
| `OriginCity` | Origin city | City associated with the landlord’s origin / base. |
| `OriginCountry` | Origin country | Country associated with the landlord’s origin / base. |
| `PortfolioSize` | Matched tenant count | How many tenant companies were matched for this landlord in the data. |
| `PortfolioActiveRate` | Share of active tenants | Fraction of matched tenants labelled as currently active / seeking work. |
| `PortfolioMeanBudget` | Average tenant monthly budget | Mean monthly budget across the landlord’s matched tenant companies. |
| `PortfolioStdBudget` | Budget variation across tenants | How much tenant monthly budgets vary (standard deviation). |
| `PortfolioMeanSales` | Average tenant sales intensity | Mean sales-of-main-product signal across tenant companies. |
| `PortfolioMeanClients` | Average tenant client base | Mean number of active clients that tenant companies report. |
| `PortfolioMeanRetention` | Average tenant client retention | Mean share of returning clients among tenants (loyalty / stickiness). |
| `PortfolioMeanSuccessScore` | Average tenant success score | Mean composite company-success score across the tenant portfolio. |
| `PortfolioStdSuccessScore` | Success-score variation across tenants | How much tenant success scores vary within the portfolio. |

## Global feature importance (mean |SHAP|)

| Meaning | Column | mean |SHAP| |
| --- | --- | ---: |
| Average tenant client base | `PortfolioMeanClients` | 0.01320 |
| Average tenant monthly budget | `PortfolioMeanBudget` | 0.00803 |
| Share of active tenants | `PortfolioActiveRate` | 0.00745 |
| Average tenant sales intensity | `PortfolioMeanSales` | 0.00621 |
| Average tenant client retention | `PortfolioMeanRetention` | 0.00606 |
| Average tenant success score | `PortfolioMeanSuccessScore` | 0.00381 |
| Matched tenant count | `PortfolioSize` | 0.00201 |
| Reported active companies | `ActiveCompanies` | 0.00196 |
| Budget variation across tenants | `PortfolioStdBudget` | 0.00129 |
| Success-score variation across tenants | `PortfolioStdSuccessScore` | 0.00119 |
| Origin country | `OriginCountry` | 0.00067 |
| Origin city | `OriginCity` | 0.00065 |
| Landlord age (years) | `LandlordAge` | 0.00061 |
| Preferred industry | `PreferredIndustry` | 0.00058 |
| Area per company | `AreaPerCompany` | 0.00023 |
| Population per company | `PopulationPerCompany` | 0.00020 |
| Property / campus area | `Area` | 0.00011 |
| Nearby population | `TotalPopulationAround` | 0.00010 |
| Population density | `PopulationDensity` | 0.00007 |

![Global importance](figures/shap/shap_global_importance.png)

## Beeswarm (direction of effects)

![Beeswarm](figures/shap/shap_beeswarm.png)

## Dependence plots

### Average tenant client base

_Column: `PortfolioMeanClients`_

![PortfolioMeanClients](figures/shap/shap_dependence_PortfolioMeanClients.png)

### Average tenant monthly budget

_Column: `PortfolioMeanBudget`_

![PortfolioMeanBudget](figures/shap/shap_dependence_PortfolioMeanBudget.png)

### Share of active tenants

_Column: `PortfolioActiveRate`_

![PortfolioActiveRate](figures/shap/shap_dependence_PortfolioActiveRate.png)

## Case studies

### 1. [GOOD] `LANDLORD_5976`

- AdjustedScore: **0.5467** · OOF pred: **0.4426** · Tenants: **23**

Landlord `LANDLORD_5976` is a **good** case with historical AdjustedScore 0.5467, and the model predicted 0.4426 (based on 23 matched tenants). What pushed the score up: Average tenant client base was 10.9 (+0.0275); Average tenant sales intensity was 0.3793 (+0.0127); Reported active companies was 29 (+0.0076). What pulled the score down: Average tenant client retention was 71.5% (-0.0134); Average tenant monthly budget was 6,229 (-0.0064). These are model associations used to explain the prediction — not proof that the landlord caused those tenant outcomes.

**What raised the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant client base | 10.9 | +0.0275 | Average tenant client base was 10.9, which increased the predicted landlord quality score by 0.0275. (Mean number of active clients that tenant companies report.) |
| Average tenant sales intensity | 0.3793 | +0.0127 | Average tenant sales intensity was 0.3793, which increased the predicted landlord quality score by 0.0127. (Mean sales-of-main-product signal across tenant companies.) |
| Reported active companies | 29 | +0.0076 | Reported active companies was 29, which increased the predicted landlord quality score by 0.0076. (Number of companies the landlord reports as active tenants.) |
| Matched tenant count | 29 | +0.0074 | Matched tenant count was 29, which increased the predicted landlord quality score by 0.0074. (How many tenant companies were matched for this landlord in the data.) |
| Budget variation across tenants | 13,847 | +0.0073 | Budget variation across tenants was 13,847, which increased the predicted landlord quality score by 0.0073. (How much tenant monthly budgets vary (standard deviation).) |

**What lowered the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant client retention | 71.5% | -0.0134 | Average tenant client retention was 71.5%, which decreased the predicted landlord quality score by 0.0134. (Mean share of returning clients among tenants (loyalty / stickiness).) |
| Average tenant monthly budget | 6,229 | -0.0064 | Average tenant monthly budget was 6,229, which decreased the predicted landlord quality score by 0.0064. (Mean monthly budget across the landlord’s matched tenant companies.) |

![Waterfall LANDLORD_5976](figures/shap/case_studies/waterfall_good_LANDLORD_5976.png)

### 2. [GOOD] `LANDLORD_8695`

- AdjustedScore: **0.5369** · OOF pred: **0.4450** · Tenants: **25**

Landlord `LANDLORD_8695` is a **good** case with historical AdjustedScore 0.5369, and the model predicted 0.4450 (based on 25 matched tenants). What pushed the score up: Average tenant client base was 16.3 (+0.0246); Average tenant sales intensity was 0.2564 (+0.0115); Matched tenant count was 39 (+0.0097). What pulled the score down: Average tenant client retention was 72.4% (-0.0138). These are model associations used to explain the prediction — not proof that the landlord caused those tenant outcomes.

**What raised the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant client base | 16.3 | +0.0246 | Average tenant client base was 16.3, which increased the predicted landlord quality score by 0.0246. (Mean number of active clients that tenant companies report.) |
| Average tenant sales intensity | 0.2564 | +0.0115 | Average tenant sales intensity was 0.2564, which increased the predicted landlord quality score by 0.0115. (Mean sales-of-main-product signal across tenant companies.) |
| Matched tenant count | 39 | +0.0097 | Matched tenant count was 39, which increased the predicted landlord quality score by 0.0097. (How many tenant companies were matched for this landlord in the data.) |
| Reported active companies | 39 | +0.0091 | Reported active companies was 39, which increased the predicted landlord quality score by 0.0091. (Number of companies the landlord reports as active tenants.) |
| Share of active tenants | 100.0% | +0.0045 | Share of active tenants was 100.0%, which increased the predicted landlord quality score by 0.0045. (Fraction of matched tenants labelled as currently active / seeking work.) |

**What lowered the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant client retention | 72.4% | -0.0138 | Average tenant client retention was 72.4%, which decreased the predicted landlord quality score by 0.0138. (Mean share of returning clients among tenants (loyalty / stickiness).) |

![Waterfall LANDLORD_8695](figures/shap/case_studies/waterfall_good_LANDLORD_8695.png)

### 3. [BAD] `LANDLORD_5634`

- AdjustedScore: **0.2484** · OOF pred: **0.2970** · Tenants: **25**

Landlord `LANDLORD_5634` is a **bad** case with historical AdjustedScore 0.2484, and the model predicted 0.2970 (based on 25 matched tenants). What pushed the score up: Nearby population was 841,693.2 (+0.0003); Landlord age (years) was 8 (+0.0002). What pulled the score down: Average tenant monthly budget was 1,813,555 (-0.0382); Average tenant sales intensity was 20.04 (-0.0282); Average tenant client base was 173.5 (-0.0218). These are model associations used to explain the prediction — not proof that the landlord caused those tenant outcomes.

**What raised the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Nearby population | 841,693.2 | +0.0003 | Nearby population was 841,693.2, which increased the predicted landlord quality score by 0.0003. (Population living around the landlord’s location.) |
| Landlord age (years) | 8 | +0.0002 | Landlord age (years) was 8, which increased the predicted landlord quality score by 0.0002. (How long the landlord has been operating (years since founding).) |

**What lowered the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant monthly budget | 1,813,555 | -0.0382 | Average tenant monthly budget was 1,813,555, which decreased the predicted landlord quality score by 0.0382. (Mean monthly budget across the landlord’s matched tenant companies.) |
| Average tenant sales intensity | 20.04 | -0.0282 | Average tenant sales intensity was 20.04, which decreased the predicted landlord quality score by 0.0282. (Mean sales-of-main-product signal across tenant companies.) |
| Average tenant client base | 173.5 | -0.0218 | Average tenant client base was 173.5, which decreased the predicted landlord quality score by 0.0218. (Mean number of active clients that tenant companies report.) |
| Share of active tenants | 96.0% | -0.0171 | Share of active tenants was 96.0%, which decreased the predicted landlord quality score by 0.0171. (Fraction of matched tenants labelled as currently active / seeking work.) |
| Budget variation across tenants | 6,212,783 | -0.0155 | Budget variation across tenants was 6,212,783, which decreased the predicted landlord quality score by 0.0155. (How much tenant monthly budgets vary (standard deviation).) |

![Waterfall LANDLORD_5634](figures/shap/case_studies/waterfall_bad_LANDLORD_5634.png)

### 4. [BAD] `LANDLORD_10894`

- AdjustedScore: **0.2224** · OOF pred: **0.3546** · Tenants: **6**

Landlord `LANDLORD_10894` is a **bad** case with historical AdjustedScore 0.2224, and the model predicted 0.3546 (based on 6 matched tenants). What pushed the score up: Average tenant client base was 24.8 (+0.0114); Average tenant client retention was 22.8% (+0.0022). What pulled the score down: Share of active tenants was 33.3% (-0.1288); Landlord age (years) was 22 (-0.0186); Average tenant sales intensity was 21.50 (-0.0186). These are model associations used to explain the prediction — not proof that the landlord caused those tenant outcomes.

**What raised the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant client base | 24.8 | +0.0114 | Average tenant client base was 24.8, which increased the predicted landlord quality score by 0.0114. (Mean number of active clients that tenant companies report.) |
| Average tenant client retention | 22.8% | +0.0022 | Average tenant client retention was 22.8%, which increased the predicted landlord quality score by 0.0022. (Mean share of returning clients among tenants (loyalty / stickiness).) |

**What lowered the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Share of active tenants | 33.3% | -0.1288 | Share of active tenants was 33.3%, which decreased the predicted landlord quality score by 0.1288. (Fraction of matched tenants labelled as currently active / seeking work.) |
| Landlord age (years) | 22 | -0.0186 | Landlord age (years) was 22, which decreased the predicted landlord quality score by 0.0186. (How long the landlord has been operating (years since founding).) |
| Average tenant sales intensity | 21.50 | -0.0186 | Average tenant sales intensity was 21.50, which decreased the predicted landlord quality score by 0.0186. (Mean sales-of-main-product signal across tenant companies.) |
| Average tenant monthly budget | 4,421 | -0.0079 | Average tenant monthly budget was 4,421, which decreased the predicted landlord quality score by 0.0079. (Mean monthly budget across the landlord’s matched tenant companies.) |
| Reported active companies | 11 | -0.0051 | Reported active companies was 11, which decreased the predicted landlord quality score by 0.0051. (Number of companies the landlord reports as active tenants.) |

![Waterfall LANDLORD_10894](figures/shap/case_studies/waterfall_bad_LANDLORD_10894.png)

### 5. [NEUTRAL] `LANDLORD_3491`

- AdjustedScore: **0.4382** · OOF pred: **0.3883** · Tenants: **5**

Landlord `LANDLORD_3491` is a **neutral** case with historical AdjustedScore 0.4382, and the model predicted 0.3883 (based on 5 matched tenants). What pushed the score up: Average tenant sales intensity was 1.833 (+0.0049); Share of active tenants was 100.0% (+0.0039); Average tenant monthly budget was 760.40 (+0.0029). What pulled the score down: Average tenant success score was 58.2% (-0.0022); Reported active companies was 6 (-0.0012); Matched tenant count was 6 (-0.0007). These are model associations used to explain the prediction — not proof that the landlord caused those tenant outcomes.

**What raised the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant sales intensity | 1.833 | +0.0049 | Average tenant sales intensity was 1.833, which increased the predicted landlord quality score by 0.0049. (Mean sales-of-main-product signal across tenant companies.) |
| Share of active tenants | 100.0% | +0.0039 | Share of active tenants was 100.0%, which increased the predicted landlord quality score by 0.0039. (Fraction of matched tenants labelled as currently active / seeking work.) |
| Average tenant monthly budget | 760.40 | +0.0029 | Average tenant monthly budget was 760.40, which increased the predicted landlord quality score by 0.0029. (Mean monthly budget across the landlord’s matched tenant companies.) |
| Preferred industry | Consumer Durables | +0.0012 | Preferred industry was Consumer Durables, which increased the predicted landlord quality score by 0.0012. (Industry focus the landlord prefers for tenants.) |
| Average tenant client retention | 41.9% | +0.0008 | Average tenant client retention was 41.9%, which increased the predicted landlord quality score by 0.0008. (Mean share of returning clients among tenants (loyalty / stickiness).) |

**What lowered the score**

| Meaning | Value | Effect (SHAP) | Explanation |
| --- | --- | ---: | --- |
| Average tenant success score | 58.2% | -0.0022 | Average tenant success score was 58.2%, which decreased the predicted landlord quality score by 0.0022. (Mean composite company-success score across the tenant portfolio.) |
| Reported active companies | 6 | -0.0012 | Reported active companies was 6, which decreased the predicted landlord quality score by 0.0012. (Number of companies the landlord reports as active tenants.) |
| Matched tenant count | 6 | -0.0007 | Matched tenant count was 6, which decreased the predicted landlord quality score by 0.0007. (How many tenant companies were matched for this landlord in the data.) |
| Nearby population | 167,379.2 | -0.0001 | Nearby population was 167,379.2, which decreased the predicted landlord quality score by 0.0001. (Population living around the landlord’s location.) |

![Waterfall LANDLORD_3491](figures/shap/case_studies/waterfall_neutral_LANDLORD_3491.png)

---
_Use wording such as “associated with” or “contributed to the prediction”; avoid causal claims unless the study design supports them._
