# Exploratory Data Analysis Report

_Generated at: 2026-07-11T09:40:31.752348+00:00_

Raw inputs: `data/raw/Landlords.parquet`, `data/raw/Companies.parquet`

## Findings (review before fixes)

### 1. [INFO] AllCompanyID parsing

AllCompanyID is stored as Python list reprs; auto-parser (ast.literal_eval) handled them and bridge IDs are clean.

```json
{
  "samples": [
    "['COMPANY_26982', 'COMPANY_2232', 'COMPANY_26983', 'COMPANY_26984', 'COMPANY_7401', 'COMPANY_25902', 'COMPANY_25903', 'C",
    "['COMPANY_16197', 'COMPANY_20308', 'COMPANY_26985', 'COMPANY_26986']",
    "['COMPANY_0548', 'COMPANY_0597', 'COMPANY_26987', 'COMPANY_1148', 'COMPANY_1212', 'COMPANY_2277', 'COMPANY_2538', 'COMPA",
    "['COMPANY_26991', 'COMPANY_0561', 'COMPANY_26992', 'COMPANY_5556', 'COMPANY_16188', 'COMPANY_8045', 'COMPANY_26010', 'CO",
    "['COMPANY_26993', 'COMPANY_26994', 'COMPANY_0855', 'COMPANY_26995', 'COMPANY_26996', 'COMPANY_26997', 'COMPANY_7695', 'C"
  ]
}
```

### 2. [WARNING] Bridge coverage

69.53% of bridge CompanyIDs match Companies.parquet (26841/38606). 11765 unmatched in bridge; 140 companies never linked. Unmatched IDs above the Companies table max are a data gap, not a parser bug.

```json
{
  "companies_in_table": 26981,
  "companies_in_bridge": 38606,
  "overlap": 26841,
  "bridge_not_in_companies": 11765,
  "companies_not_in_bridge": 140,
  "match_rate_bridge_pct": 69.53,
  "coverage_rate_companies_pct": 99.48,
  "malformed_bridge_ids": 0,
  "malformed_examples": [
    "COMPANY_26982",
    "COMPANY_26983",
    "COMPANY_26984",
    "COMPANY_26985",
    "COMPANY_26986",
    "COMPANY_26987",
    "COMPANY_26988",
    "COMPANY_26989"
  ],
  "orphan_examples": [
    "COMPANY_0222",
    "COMPANY_10154",
    "COMPANY_10175",
    "COMPANY_10205",
    "COMPANY_11135",
    "COMPANY_11214",
    "COMPANY_11343",
    "COMPANY_11434"
  ]
}
```

### 3. [WARNING] Companies missingness

10 company columns have >=40% null (MonthlyBudget, Rank, YearFounded, CompanyStatus, ClientsInTheLast7Days, ClientsInTheLast6Months, ClientsInTheLast12Months, TotalClientsInTheLast7Days...).

```json
{
  "MonthlyBudget": {
    "null_pct": 80.17,
    "n_unique": 3103
  },
  "Rank": {
    "null_pct": 86.05,
    "n_unique": 3137
  },
  "YearFounded": {
    "null_pct": 55.37,
    "n_unique": 209
  },
  "CompanyStatus": {
    "null_pct": 47.95,
    "n_unique": 8
  },
  "ClientsInTheLast7Days": {
    "null_pct": 98.15,
    "n_unique": 6
  },
  "ClientsInTheLast6Months": {
    "null_pct": 73.4,
    "n_unique": 43
  },
  "ClientsInTheLast12Months": {
    "null_pct": 58.49,
    "n_unique": 63
  },
  "TotalClientsInTheLast7Days": {
    "null_pct": 96.61,
    "n_unique": 7
  },
  "TotalClientsInTheLast6Months": {
    "null_pct": 62.04,
    "n_unique": 78
  },
  "TotalClientsInTheLast12Months": {
    "null_pct": 46.84,
    "n_unique": 113
  }
}
```

### 4. [WARNING] CompanyStatus label quality

CompanyStatus is sparse/ambiguous (null~=48.0%, literal 'None'~=48.0%). Binary good/bad mapping will be noisy.

```json
{
  "null_pct": 47.95,
  "none_pct": 47.95,
  "value_counts": {
    "Actively Seeking New Employees": 13315,
    "null": 12938,
    "Out of Business": 364,
    "Acquired/Merged (Operating Subsidiary)": 194,
    "Acquired/Merged": 73,
    "Not Making New Products": 47,
    "Other": 24,
    "Will Consider New Projects": 18,
    "Reducing Activity": 8
  }
}
```

### 5. [INFO] Zero-inflated sales

SalesOfMainProduct is 82.8% zeros - consider log1p / tree models.

```json
{
  "zero_pct": 82.76
}
```

### 6. [INFO] Zero-inflated sales

SalesOfOtherProduct is 61.4% zeros - consider log1p / tree models.

```json
{
  "zero_pct": 61.38
}
```

### 7. [INFO] ActiveCompanies consistency

ActiveCompanies aligns with parsed portfolio size for 98.94% of landlords.

```json
{
  "exact_match_pct": 98.94,
  "within_1_pct": 99.63,
  "median_diff": 0.0,
  "mean_claimed": 9.134657398212513,
  "mean_parsed": 9.117064801372575
}
```

### 8. [INFO] Row uniqueness

No full-row duplicates in Landlords or Companies.

## Dataset shapes

| dataset | rows | cols | memory_mb | duplicate_rows |
| --- | --- | --- | --- | --- |
| Landlords | 15154 | 10 | 8.82 | 0 |
| Companies | 26981 | 20 | 11.37 | 0 |

## Identifier integrity

| dataset | id_col | present | rows | unique_ids | null_ids | duplicate_ids |
| --- | --- | --- | --- | --- | --- | --- |
| Landlords | LandLordID | True | 15154 | 15154 | 0 | 0 |
| Companies | CompanyID | True | 26981 | 26981 | 0 | 0 |

## AllCompanyID format inspection

```json
{
  "present": true,
  "sample_types": [
    "str"
  ],
  "sample_values": [
    "['COMPANY_26982', 'COMPANY_2232', 'COMPANY_26983', 'COMPANY_26984', 'COMPANY_7401', 'COMPANY_25902', 'COMPANY_25903', 'C",
    "['COMPANY_16197', 'COMPANY_20308', 'COMPANY_26985', 'COMPANY_26986']",
    "['COMPANY_0548', 'COMPANY_0597', 'COMPANY_26987', 'COMPANY_1148', 'COMPANY_1212', 'COMPANY_2277', 'COMPANY_2538', 'COMPA",
    "['COMPANY_26991', 'COMPANY_0561', 'COMPANY_26992', 'COMPANY_5556', 'COMPANY_16188', 'COMPANY_8045', 'COMPANY_26010', 'CO",
    "['COMPANY_26993', 'COMPANY_26994', 'COMPANY_0855', 'COMPANY_26995', 'COMPANY_26996', 'COMPANY_26997', 'COMPANY_7695', 'C"
  ],
  "of_first_n": 200,
  "already_list": 0,
  "looks_like_python_list_repr": 200,
  "looks_like_json": 0,
  "looks_like_csv": 0
}
```

## Column profiles — Landlords

| column | dtype | non_null | null_pct | n_unique | top_value |
| --- | --- | --- | --- | --- | --- |
| LandLordID | object | 15154 | 0.0 | 15154 | LANDLORD_0001 |
| YearFounded | float64 | 14923 | 1.52 | 60 | 2020.0 |
| PreferredIndustry | object | 15154 | 0.0 | 40 | Software |
| LandlordOriginCity | object | 15108 | 0.3 | 1862 | San Francisco |
| LandlordOriginCountry | object | 15154 | 0.0 | 64 | United States |
| ActiveCompanies | float64 | 15105 | 0.32 | 80 | 5.0 |
| TotalPopulationAround | float64 | 15154 | 0.0 | 15154 | 21626.62143 |
| Area | int64 | 15154 | 0.0 | 94 | 30 |
| AllCompanyID | object | 15154 | 0.0 | 14990 | ['COMPANY_0335'] |
| LastUpdated | object | 15154 | 0.0 | 5 | 02/22/2026 |

## Column profiles — Companies

| column | dtype | non_null | null_pct | n_unique | top_value |
| --- | --- | --- | --- | --- | --- |
| CompanyID | object | 26981 | 0.0 | 26981 | COMPANY_0001 |
| SalesOfMainProduct | int64 | 26981 | 0.0 | 87 | 0 |
| SalesOfOtherProduct | int64 | 26981 | 0.0 | 169 | 0 |
| MonthlyBudget | float64 | 5350 | 80.17 | 3103 | 100.0 |
| PrimaryType | object | 26981 | 0.0 | 22 | Angel (individual) |
| Rank | float64 | 3764 | 86.05 | 3137 | 0.0 |
| YearFounded | float64 | 12041 | 55.37 | 209 | 2020.0 |
| CompanyStatus | object | 14043 | 47.95 | 8 | Actively Seeking New Employees |
| OriginCity | object | 18649 | 30.88 | 2190 | New York |
| OriginCountry | object | 19000 | 29.58 | 106 | United States |
| Investments | int64 | 26981 | 0.0 | 148 | 1 |
| TodaysClients | int64 | 26981 | 0.0 | 535 | 1 |
| ReturningClient | float64 | 26739 | 0.9 | 121 | 1.0 |
| TotalActiveClients | float64 | 26857 | 0.46 | 307 | 1.0 |
| ClientsInTheLast7Days | float64 | 500 | 98.15 | 6 | 1.0 |
| ClientsInTheLast6Months | float64 | 7176 | 73.4 | 43 | 1.0 |
| ClientsInTheLast12Months | float64 | 11200 | 58.49 | 63 | 1.0 |
| TotalClientsInTheLast7Days | float64 | 914 | 96.61 | 7 | 1.0 |
| TotalClientsInTheLast6Months | float64 | 10242 | 62.04 | 78 | 1.0 |
| TotalClientsInTheLast12Months | float64 | 14342 | 46.84 | 113 | 1.0 |

## Numeric summary — Landlords

| column | count | mean | std | min | 25% | 50% | 75% | max | skew | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YearFounded | 14923.0 | 2017.862 | 5.3047 | 1852.0 | 2016.0 | 2019.0 | 2021.0 | 2026.0 | -5.3712 | 99.137 |
| ActiveCompanies | 15105.0 | 9.1347 | 8.0255 | 1.0 | 4.0 | 7.0 | 12.0 | 134.0 | 2.8217 | 17.6125 |
| TotalPopulationAround | 15154.0 | 498795.4082 | 288282.2913 | 33.2725 | 248853.0789 | 496191.768 | 748240.5728 | 999980.5033 | 0.017 | -1.1928 |
| Area | 15154.0 | 39.346 | 23.6321 | 0.0 | 20.0 | 39.0 | 59.0 | 750.0 | 2.0745 | 55.7798 |

## Numeric summary — Companies

| column | count | mean | std | min | 25% | 50% | 75% | max | skew | kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SalesOfMainProduct | 26981.0 | 0.8798 | 5.0037 | 0.0 | 0.0 | 0.0 | 0.0 | 176.0 | 14.6004 | 307.4803 |
| SalesOfOtherProduct | 26981.0 | 3.4084 | 14.4585 | 0.0 | 0.0 | 0.0 | 2.0 | 633.0 | 15.7818 | 415.4365 |
| MonthlyBudget | 5350.0 | 61648.0735 | 895671.1313 | 0.0 | 67.72 | 286.1886 | 2000.0 | 43000000.0 | 34.2519 | 1364.7745 |
| Rank | 3764.0 | 658.2262 | 5541.1073 | 0.0 | 4.0426 | 31.189 | 141.0 | 252367.2991 | 30.999 | 1252.988 |
| YearFounded | 12041.0 | 2001.5818 | 32.301 | 1805.0 | 2001.0 | 2013.0 | 2019.0 | 2026.0 | -2.8266 | 8.7958 |
| Investments | 26981.0 | 4.2706 | 13.0312 | 1.0 | 1.0 | 1.0 | 3.0 | 632.0 | 16.6058 | 506.1193 |
| TodaysClients | 26981.0 | 26.777 | 93.2574 | 1.0 | 2.0 | 5.0 | 20.0 | 3655.0 | 15.1612 | 376.1169 |
| ReturningClient | 26739.0 | 3.4551 | 9.3301 | 1.0 | 1.0 | 1.0 | 2.0 | 531.0 | 16.6174 | 569.8724 |
| TotalActiveClients | 26857.0 | 12.6419 | 44.5549 | 1.0 | 1.0 | 3.0 | 11.0 | 2163.0 | 22.6116 | 815.9612 |
| ClientsInTheLast7Days | 500.0 | 1.112 | 0.5215 | 1.0 | 1.0 | 1.0 | 1.0 | 8.0 | 7.7966 | 80.6911 |
| ClientsInTheLast6Months | 7176.0 | 2.0258 | 3.1614 | 1.0 | 1.0 | 1.0 | 2.0 | 67.0 | 8.6012 | 110.0043 |
| ClientsInTheLast12Months | 11200.0 | 2.5668 | 4.9518 | 1.0 | 1.0 | 1.0 | 2.0 | 111.0 | 9.3682 | 132.9238 |
| TotalClientsInTheLast7Days | 914.0 | 1.1772 | 0.5876 | 1.0 | 1.0 | 1.0 | 1.0 | 9.0 | 5.6864 | 49.4134 |
| TotalClientsInTheLast6Months | 10242.0 | 3.2811 | 7.3668 | 1.0 | 1.0 | 1.0 | 3.0 | 265.0 | 15.1837 | 392.0868 |
| TotalClientsInTheLast12Months | 14342.0 | 4.5694 | 12.0372 | 1.0 | 1.0 | 2.0 | 4.0 | 538.0 | 17.7949 | 539.1737 |

## Category frequencies — Landlords

| column | value | count | pct |
| --- | --- | --- | --- |
| PreferredIndustry | Software | 6294 | 41.53 |
| PreferredIndustry | Commercial Services | 1640 | 10.82 |
| PreferredIndustry | Healthcare Technology Systems | 694 | 4.58 |
| PreferredIndustry | Media | 605 | 3.99 |
| PreferredIndustry | Consumer Non-Durables | 598 | 3.95 |
| PreferredIndustry | Commercial Products | 538 | 3.55 |
| PreferredIndustry | Services (Non-Financial) | 466 | 3.08 |
| PreferredIndustry | Computer Hardware | 408 | 2.69 |
| PreferredIndustry | Healthcare Devices and Supplies | 408 | 2.69 |
| PreferredIndustry | Consumer Durables | 388 | 2.56 |
| LastUpdated | 02/22/2026 | 6226 | 41.08 |
| LastUpdated | 02/02/2026 | 5119 | 33.78 |
| LastUpdated | 02/15/2026 | 2567 | 16.94 |
| LastUpdated | 02/08/2026 | 641 | 4.23 |
| LastUpdated | 02/09/2026 | 601 | 3.97 |

## Category frequencies — Companies

| column | value | count | pct |
| --- | --- | --- | --- |
| PrimaryType | Angel (individual) | 9165 | 33.97 |
| PrimaryType | Pets | 6916 | 25.63 |
| PrimaryType | Banking | 3486 | 12.92 |
| PrimaryType | Other | 1182 | 4.38 |
| PrimaryType | Alcohol | 1049 | 3.89 |
| PrimaryType | Toys | 933 | 3.46 |
| PrimaryType | Office | 646 | 2.39 |
| PrimaryType | Growth/Expansion | 429 | 1.59 |
| PrimaryType | Angel Group | 415 | 1.54 |
| PrimaryType | Family Office | 398 | 1.48 |
| CompanyStatus | Actively Seeking New Employees | 13315 | 49.35 |
| CompanyStatus | None | 12938 | 47.95 |
| CompanyStatus | Out of Business | 364 | 1.35 |
| CompanyStatus | Acquired/Merged (Operating Subsidiary) | 194 | 0.72 |
| CompanyStatus | Acquired/Merged | 73 | 0.27 |
| CompanyStatus | Not Making New Products | 47 | 0.17 |
| CompanyStatus | Other | 24 | 0.09 |
| CompanyStatus | Will Consider New Projects | 18 | 0.07 |
| CompanyStatus | Reducing Activity | 8 | 0.03 |

## Bridge & portfolio

```json
{
  "portfolio_size_stats": {
    "count": 15154.0,
    "mean": 9.12,
    "std": 8.02,
    "min": 1.0,
    "25%": 4.0,
    "50%": 7.0,
    "75%": 12.0,
    "max": 134.0
  },
  "landlords_with_single_company": 1077,
  "total_landlords": 15154,
  "total_links": 138160,
  "unique_companies": 38606,
  "portfolio_buckets": {
    "1": 1077,
    "2": 1195,
    "3": 1230,
    "4": 1167,
    "5-9": 5044,
    "10-19": 4082,
    "20-49": 1305,
    "50-99": 52,
    "100+": 2
  },
  "coverage": {
    "companies_in_table": 26981,
    "companies_in_bridge": 38606,
    "overlap": 26841,
    "bridge_not_in_companies": 11765,
    "companies_not_in_bridge": 140,
    "match_rate_bridge_pct": 69.53,
    "coverage_rate_companies_pct": 99.48,
    "malformed_bridge_ids": 0,
    "malformed_examples": [
      "COMPANY_26982",
      "COMPANY_26983",
      "COMPANY_26984",
      "COMPANY_26985",
      "COMPANY_26986",
      "COMPANY_26987",
      "COMPANY_26988",
      "COMPANY_26989"
    ],
    "orphan_examples": [
      "COMPANY_0222",
      "COMPANY_10154",
      "COMPANY_10175",
      "COMPANY_10205",
      "COMPANY_11135",
      "COMPANY_11214",
      "COMPANY_11343",
      "COMPANY_11434"
    ]
  },
  "active_vs_parsed": {
    "exact_match_pct": 98.94,
    "within_1_pct": 99.63,
    "median_diff": 0.0,
    "mean_claimed": 9.134657398212513,
    "mean_parsed": 9.117064801372575
  }
}
```

### Top landlords by portfolio size

| LandLordID | n_companies |
| --- | --- |
| LANDLORD_11865 | 134 |
| LANDLORD_11297 | 125 |
| LANDLORD_11884 | 95 |
| LANDLORD_12034 | 89 |
| LANDLORD_7813 | 86 |
| LANDLORD_12325 | 83 |
| LANDLORD_6947 | 82 |
| LANDLORD_4373 | 81 |
| LANDLORD_3737 | 80 |
| LANDLORD_13604 | 79 |

## Top correlations

### Landlords
| a | b | abs_corr |
| --- | --- | --- |
| YearFounded | ActiveCompanies | 0.0239 |
| YearFounded | TotalPopulationAround | 0.0135 |
| TotalPopulationAround | Area | 0.003 |
| YearFounded | Area | 0.0026 |
| ActiveCompanies | TotalPopulationAround | 0.0023 |
| ActiveCompanies | Area | 0.0006 |

### Companies
| a | b | abs_corr |
| --- | --- | --- |
| Investments | ReturningClient | 0.9845 |
| ClientsInTheLast6Months | ClientsInTheLast12Months | 0.9543 |
| TotalClientsInTheLast6Months | TotalClientsInTheLast12Months | 0.9542 |
| Investments | ClientsInTheLast12Months | 0.9018 |
| TodaysClients | TotalActiveClients | 0.875 |
| ReturningClient | ClientsInTheLast12Months | 0.8726 |
| ClientsInTheLast7Days | TotalClientsInTheLast7Days | 0.8428 |
| Investments | ClientsInTheLast6Months | 0.8327 |
| SalesOfOtherProduct | TodaysClients | 0.8033 |
| ReturningClient | ClientsInTheLast6Months | 0.7995 |
| TodaysClients | TotalClientsInTheLast12Months | 0.7639 |
| TotalActiveClients | TotalClientsInTheLast12Months | 0.7563 |
| TodaysClients | TotalClientsInTheLast6Months | 0.7152 |
| ClientsInTheLast7Days | ClientsInTheLast6Months | 0.7003 |
| SalesOfMainProduct | SalesOfOtherProduct | 0.6877 |

## Figures

- `reports/eda_runs/20260711_134026/figures/portfolio_size.png`

![figure](reports/eda_runs/20260711_134026/figures/portfolio_size.png)

- `reports/eda_runs/20260711_134026/figures/landlords_numeric.png`

![figure](reports/eda_runs/20260711_134026/figures/landlords_numeric.png)

- `reports/eda_runs/20260711_134026/figures/companies_numeric.png`

![figure](reports/eda_runs/20260711_134026/figures/companies_numeric.png)

- `reports/eda_runs/20260711_134026/figures/landlords_categories.png`

![figure](reports/eda_runs/20260711_134026/figures/landlords_categories.png)

- `reports/eda_runs/20260711_134026/figures/companies_categories.png`

![figure](reports/eda_runs/20260711_134026/figures/companies_categories.png)

- `reports/eda_runs/20260711_134026/figures/companies_correlation.png`

![figure](reports/eda_runs/20260711_134026/figures/companies_correlation.png)

---
_Issues listed above are intentional for review; do not treat this report as a fix log._
