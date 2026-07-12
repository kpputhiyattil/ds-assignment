# Mini Report: LLM-Powered Credit Assessment Agent

This report answers the assignment questions on **guardrail design**, **evaluation at scale**, and **production monitoring**, and records **example assessments** from the running system. It also lists **improvements needed** before production use.

---

## 1. Guardrail Design — How Were Guardrails and Thresholds Chosen?

Hard rules run **after** the LLM verdict and read **deterministic tool outputs only** (never the LLM rationale). The first firing rule wins. Implementation: `agent/guardrails.py`.

### Guardrail 1 — Company Status Hard Stop

| | |
|---|---|
| **Rule** | If `business_profile.status_flag` ∈ {`inactive`, `suspended`, `closed`} → force `No-Go`, `confidence=high` |
| **Threshold** | Binary (active vs non-active). No grey zone. |
| **Why** | A non-operating company cannot use or repay growth capital. Engagement or financial trends cannot override that. An LLM that recommends `Continue` for a closed company is a critical failure that must be suppressed, not softened. |
| **Auditability** | Reads `status_flag` from `check_business_profile`, not free-text reasoning. |

Unknown status does **not** hard-stop today; it is logged and left to the LLM / Review path (see Improvements).

### Guardrail 2 — Extreme Budget Burn (financial hard stop)

| | |
|---|---|
| **Rule** | If `financial_health.budget_utilisation_ratio < 0.10` → force `No-Go` |
| **Threshold** | `BUDGET_BURN_THRESHOLD = 0.10` (tunable constant) |
| **Why this cut** | Among rows with `MonthlyBudget > 0`, utilisation is heavily left-skewed (median ≈ 1%). Cuts at 30% / 40% / 50% would flag ~92–94% of that subset and are too blunt as a **hard** override. **10%** is a catastrophic-coverage stop: still a clear lending red flag, while **missing** ratios do **not** fire (no penalty for incomplete data). |
| **Why needed** | Strong retention/growth can sway the LLM while unit economics are insolvent. This meets the assignment requirement for at least one hard **financial** guardrail. |

### Soft calibration (not a guardrail)

Before hard guardrails, a **Continue calibration** step (`_promote_continue_if_eligible` in `credit_agent.py`) may promote an overly cautious LLM `Review`/`No-Go` to `Continue` when tools show an active company clearing a first-pass bar (util ≥ 0.10 and retention ≥ 0.15, with stronger util/retention combinations). Hard guardrails still win afterward.

---

## 2. Evaluation — How Would You Know the Agent Is Making Good Decisions at Scale?

| Method | What to measure | Decision rule |
|---|---|---|
| **Offline labelled set** | Sample 50–100 companies; senior analysts label `Continue` / `Review` / `No-Go`. Run the agent; report precision, recall, F1 **per class**. | Prioritise false positives on `Continue` (approving a bad company costs more than rejecting a good one). |
| **Analyst agreement** | Share of agent verdicts overridden by analysts. | Sustained override rate **>20%** → systematic mismatch in signal weights. |
| **Rationale quality (LLM-as-judge)** | Separate GPT-4o call with a rubric: factual accuracy, three-tool coverage, coherence (1–5). | Flag scores **&lt;3** for human review; track mean score over time. |
| **Guardrail fire rate** | % of assessments with `guardrail_fired=true`, split by rule. | Spike → data-quality issue; prolonged **zero** → threshold may be misaligned. |
| **Outcome A/B (production)** | 90-day default / delinquency rates for agent-assisted vs analyst-only approvals. | Gold-standard credit quality metric. |

**Batch harness already in place:** `python scripts/batch_assess.py` (size via `BATCH_SIZE` in `.env`) writes verdict distributions and ID lists under `outputs/`, which can feed offline evaluation and fire-rate dashboards.

### Observed batch snapshot (n=150)

| Recommendation | Count | Share |
|---|---|---|
| Continue | 2 | 1.3% |
| Review | 43 | 28.7% |
| No-Go | 105 | 70.0% |
| Errors | 0 | 0% |

- Guardrails fired on **87 / 150** (~58%): mostly `extreme_budget_burn`, a few `company_status_hard_stop`.
- Model in that run: configured OpenAI-compatible endpoint (see batch JSON metadata).
- Interpretation: the dataset is financially stressed; a high No-Go / guardrail rate is expected with a 10% burn threshold, but Continue is rare — calibration and prompt tuning remain open (see §5).

---

## 3. Production Monitoring — Drift and Untrustworthy Outputs

| Signal | Source | Alert condition |
|---|---|---|
| Recommendation mix | Langfuse + dashboard | **>20%** week-over-week shift in Continue / Review / No-Go share |
| Guardrail override rate | Langfuse custom metric | Spike **>3σ** above 30-day baseline (by rule name) |
| Rationale length | Langfuse metadata | Persistently **&lt;50** words (shortcuts) or **>500** words (verbosity / hallucination) |
| Tool-call completeness | Langfuse trace | Assessment with **&lt;3** tool calls → prompt drift or model regression |
| LLM latency p95 | Langfuse + provider dashboard | **>10s** p95 |
| Input null rates | DataLoader / pipeline metrics | Spike in key fields vs deployment baseline |
| Analyst override rate | Application audit log | **>30%** of verdicts overridden in a rolling 7-day window |
| Cost per assessment | Provider usage API | **>2×** baseline → runaway loop or prompt injection |

### Drift detection

- **Data drift:** Weekly PSI on `budget_utilisation_ratio`, `retention_rate`, and `business_type_risk_tier` vs deployment baseline. **PSI > 0.25** on any metric → manual data-quality review.
- **Model drift:** On provider model upgrades, re-run the offline labelled set. **>5%** F1 drop on any class → roll back model / prompt version.

---

## 4. Example Assessments

Reproduce with:

```bash
python scripts/batch_assess.py --demo-three
# or pick the IDs below in Streamlit → Run Assessment
```

| CompanyID | Verdict | Guardrail | Evidence (from tools / run) |
|---|---|---|---|
| `COMPANY_0088` | **Continue** | No | Active; util ≈ 5.91×; retention ≈ 0.47. Continue calibration may promote if the LLM under-calls. |
| `COMPANY_0015` | **Review** | No | Active but material gaps (e.g. zero / missing budget → util unavailable); engagement incomplete → Review, not Continue. |
| `COMPANY_0093` | **No-Go** | **Yes — `company_status_hard_stop`** | Status `closed`. UI shows guardrail banner; rationale prefixed with `[GUARDRAIL OVERRIDE]`. |
| `COMPANY_0002` | **No-Go** | **Yes — `extreme_budget_burn`** | Util ≈ 1.7% (&lt;10%). Demonstrates the financial hard stop independently of status. |

> Note: `COMPANY_0001` is a useful **weak / incomplete financials** case (often `No-Go` from the LLM without a status guardrail). Prefer `COMPANY_0015` for a clean **Review** demo and `COMPANY_0093` / `COMPANY_0002` for the two guardrail types.

---

## 5. Improvements Needed

Prioritised gaps before treating this as production-ready:

1. **Continue rate calibration** — On the n=150 run, only **2 Continues**. Revisit prompt thresholds, Continue calibration floors, and whether the 10% burn guardrail should remain a hard stop vs a strong Review signal for borderline cases. Validate against analyst-labelled gold data.
2. **Unknown status policy** — Today `unknown` only logs a warning. Decide: hard `Review` floor, or conservative `No-Go` for lending risk appetite.
3. **Gold-labelled evaluation set** — Build and version 50–100 analyst-labelled companies; automate precision/recall in CI against a frozen model/prompt pair.
4. **Guardrail threshold governance** — Expose `BUDGET_BURN_THRESHOLD` via config (not only a module constant); document change control and backtest impact on historical batches before changing.
5. **Richer monitoring** — Wire Langfuse metrics for recommendation mix, guardrail rule rates, and tool-call counts; add PSI jobs on tool metrics.
6. **Data quality UX** — Surface `data_quality_issues` from tools more prominently in Streamlit so analysts see *why* confidence is low.
7. **Failure handling** — Harden timeouts, partial tool failures, and provider outages with clearer analyst-facing states (already partially handled; needs runbook + alerting).
8. **Cost / latency controls** — Cap retries, track tokens per assessment, and support cheaper models for triage with escalation to GPT-4o for edge cases.
9. **Human-in-the-loop workflow** — Persist assessments, capture analyst overrides, and feed overrides back into evaluation (closes the agreement-rate loop in §2).
10. **Security & compliance** — Do not log full company profiles; add access control if the Streamlit app leaves local demo use.

---

## References in this repo

| Artifact | Path |
|---|---|
| Guardrails | `agent/guardrails.py` |
| Continue calibration | `agent/credit_agent.py` (`_promote_continue_if_eligible`) |
| System prompt | `agent/prompts.py` |
| Batch runner (`BATCH_SIZE` in `.env`) | `scripts/batch_assess.py` |
| Example batch output | `outputs/batch_150_*.json` |
| How to run | `README.md` |
