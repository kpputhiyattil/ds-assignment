# LIQUIDITY DATA SCIENCE TASKS

## Submission Guidelines

**Format:** Submit a zipped folder containing your git repository. We will review your commit history as part of the evaluation, please commit incrementally as you work rather than in a single final commit.

**AI Tools:** You are encouraged to use AI coding assistants. You are fully responsible for the code and outputs you submit, be prepared to explain any part of your work and the decisions behind it.

**Language:** Python. Jupyter notebooks are acceptable; clean, runnable `.py` files are a plus.

**Mini Report:** Each task should include a short written summary (a dedicated cell or section) covering: your approach and key assumptions, what the model or analysis revealed, and what you would improve given more time.

---

## 1. "Good Landlord" vs. "Bad Landlord"

### Background

Assume there is a country where landlords can be evaluated by an AI model, and anyone who wants to buy or rent from them relies on this model.

### Business Requirement

Your task is to develop such a model for landlords and businesses. You will receive historical data of landlords, tenants, and companies, and you will need to find a method or metric that estimates whether a given landlord is "good" or "bad" for companies' success, i.e., is it worth for a business or shop to rent an area from that landlord.

### Exercise Guideline

You can use any classifier or regressor you like. Your work should cover all the standard steps of a data science project: preprocessing, feature engineering, model selection, and evaluation.

Beyond model performance, explain *why* your model makes the predictions it does. Use model interpretability techniques (such as SHAP or feature importance) to surface the most influential signals and present them clearly.

---

**`LandLords.parquet`**, Landlord data with the following fields:

- `LandLordID`, ID of the landlord
- `YearFounded`, Year of foundation
- `PreferredIndustry`, Landlord's preferred industry type
- `LandlordOriginCity`, Origin city of the landlord
- `LandlordOriginCountry`, Origin country of the landlord
- `ActiveCompanies`, Number of current renting companies
- `TotalPopulationAround`, Total population nearby the rented area
- `Area`, Total rented area
- `AllCompanyID`, IDs of the landlord's renting companies
- `LastUpdated`, Date of data update

**`Companies.parquet`**, Company data with the following fields:

- `CompanyID`, ID of the company
- `SalesOfMainProduct`, Number of sales of the main product in a given period
- `SalesOfOtherProduct`, Number of sales of other products in the same period
- `MonthlyBudget`, Company monthly budget
- `PrimaryType`, Business type
- `Rank`, Average customer rank
- `YearFounded`, Year of foundation
- `CompanyStatus`, Current business status
- `OriginCity`, Origin city
- `OriginCountry`, Origin country
- `Investments`, Number of investments received
- `TodaysClients`, Clients on the last update date
- `ReturningClient`, Number of returning clients
- `TotalActiveClients`, Number of active clients
- `ClientsInTheLast7Days`, Clients in the last 7 days
- `ClientsInTheLast6Months`, Clients in the last 6 months
- `ClientsInTheLast12Months`, Clients in the last 12 months
- `TotalVisitorsInTheLast7Days`, Visitors in the last 7 days
- `TotalVisitorsInTheLast6Months`, Visitors in the last 6 months
- `TotalVisitorsInTheLast12Months`, Visitors in the last 12 months

---

## 2. Credit Assessment Agent

### Background

Liquidity's credit team evaluates company applications for growth capital on an ongoing basis. While predictive models surface key signals automatically, the assessment process requires reasoning across multiple financial dimensions, revenue trajectory, payment regularity, business health indicators, before a human analyst can form a view. To accelerate this first-pass triage, the team wants an AI agent that can ingest a company's financial profile, reason through the relevant credit signals, and produce an explainable recommendation.

### Problem Description

The current process requires an analyst to manually aggregate signals from multiple data sources before forming an initial position. This creates bottlenecks and inconsistency across analysts. The goal is an intelligent agent that automates first-pass assessment while remaining transparent and auditable, the analyst must be able to understand *why* the agent reached its conclusion and override it when needed.

### Business Requirement

Build an LLM-powered credit assessment agent that accepts a company profile as input and produces a structured output:

- **Recommendation:** Continue / Review / No-Go
- **Confidence level:** low / medium / high
- **Natural-language rationale** summarizing the key factors that drove the decision

The agent must enforce at least one hard financial guardrail, a rule-based check that overrides the LLM's recommendation regardless of its output. Document the reasoning behind your choice of guardrail.

### Exercise Guideline

You may use any LLM provider and any agentic framework, or build from scratch. The agent should use a tool-calling pattern with at least **3 defined tools** it can invoke during its reasoning process. The specific tools are your design choice, consider what signals a credit analyst would want to compute or retrieve from the available data.

Demonstrate the agent on at least **3 company profiles**, including at least one case where the hard guardrail fires and overrides the model's recommendation. Show the agent's reasoning steps, not just the final output.

Build a **Streamlit application** to present your results. The app should allow a user to select a company, trigger the agent, and view the recommendation, confidence level, rationale, and the key signals the agent considered. You are welcome to use AI coding assistants to build the Streamlit layer, if you do, briefly document what you generated and what you adjusted.

In your mini report, address:

- How did you decide what the guardrail should be and what threshold did you set?
- How would you evaluate whether the agent is making good decisions at scale?
- What would you monitor in production to detect when the agent's outputs are drifting or becoming untrustworthy?

---

**`Companies.parquet`**, Company profiles with business health and engagement signals (same file as Task 1)

---


## 3. The Billing / Invoice Problem

### Background

A financial service provider uses a Billing System API to assess the credit worthiness of prospective customers. Data obtained from this API is referred to as "invoice data" and is used across descriptive and predictive functions within the credit model. A key attribute is the billing interval of an invoice, typically monthly or annually, with quarterly and semi-annual intervals also common. One-time billing representing setup fees is another interval type.

### Problem Description

The classification of billing interval exists in the customer's accounting or CRM system. However, due to technical difficulties, this attribute is not always available at the point of first integration. As a result, the automated credit assessment process is delayed, with a negative impact on business flow.

### Business Requirement

Provide an assessment of the invoice billing cycle that is "good enough" for an initial response to the customer ("Continue" / "No Go").

### Exercise Guideline

The dataset contains only raw invoice records, there is no billing interval column. Your task has three parts:

1. **Infer billing intervals** from patterns in invoice dates and amounts per customer (e.g. monthly, quarterly, annual, one-time). Explain your derivation logic, there is no ground truth to validate against.
2. **Build a dollar churn model**, define churn in dollar terms and predict it from invoice-derived features.
3. **Test the interval's value** by comparing the churn model with and without your inferred interval as a feature. If it improves performance, it is "good enough" to substitute for the missing real interval and unblock the Continue / No-Go decision.

---

**`Invoices_users.parquet`**, Invoice data with the following fields:

- `id`, Invoice ID
- `account`, Account type
- `accounted`, Account ID
- `customerid`, Customer ID
- `amount`, Invoice amount
- `date`, Invoice date

---