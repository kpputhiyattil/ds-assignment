# Mini Report: LLM-Powered Credit Assessment Agent

## 1. Guardrail Design

### Guardrail 1 — Company Status Hard Stop

**Rule:** If `CompanyStatus` normalises to `inactive`, `suspended`, or `closed` → override to `No-Go` with `confidence=high`.

**Threshold:** Binary. There is no grey zone — an inactive company is not a lesser version of an active one.

**Rationale:** A company that is not operationally active cannot use or repay growth capital. This is the clearest bright-line signal in the dataset. Any LLM reasoning that reaches `Continue` for a closed company is a hallucination that must be suppressed — no engagement metric or financial trend can override the fundamental inability to operate. The rule reads directly from the deterministic tool output (`business_profile.status_flag`), not from the LLM's rationale text, making it fully auditable.

### Guardrail 2 — Extreme Budget Burn

**Rule:** If `budget_utilisation_ratio < 0.10` (revenue covers <10% of monthly budget) → override to `No-Go`.

**Threshold selection:** Among rows with `MonthlyBudget > 0`, utilisation is heavily left-skewed (median ≈ 1%). Candidate cuts at 30%/40%/50% would flag ~92–94% of that subset and are too blunt as a hard override. **10%** is used as a catastrophic-coverage financial hard stop — still a clear lending red flag, while missing ratios do not fire. The constant `BUDGET_BURN_THRESHOLD` in `agent/guardrails.py` is tunable.

**Rationale:** Strong client engagement cannot compensate for catastrophic unit economics. The guardrail catches cases where the LLM is swayed by retention or growth metrics while ignoring that the company cannot service growth capital. This satisfies the assignment requirement for at least one hard financial guardrail that overrides the LLM.

---

## 2. Evaluation: How Would You Know the Agent Is Making Good Decisions at Scale?

**Offline evaluation set:** Sample 50–100 companies and have senior analysts label them (`Continue / Review / No-Go`). Run the agent on the same set. Measure precision, recall, and F1 per class. Pay particular attention to false positives on `Continue` — approving a bad company is a higher-cost error than rejecting a good one.

**Agreement rate:** Track the fraction of agent verdicts that analysts override. A sustained override rate >20% signals systematic drift between the agent's signal weights and analyst judgment.

**Rationale quality scoring:** Use a separate LLM call (GPT-4o with a rubric prompt) to score each rationale on a 1–5 scale for factual accuracy (did it cite the right numbers?), signal coverage (did it address all three tools?), and coherence (is the recommendation consistent with the evidence?). Flag rationales scoring <3.

**Guardrail fire rate:** Monitor what fraction of assessments trigger a guardrail. A spike suggests upstream data quality degradation; a persistent zero rate suggests the guardrail thresholds may have drifted out of alignment with current data.

**A/B cohort study (production):** If the agent is deployed alongside human analysts, compare 90-day default rates on loans approved by each stream. This is the gold-standard outcome metric — it directly measures credit decision quality.

---

## 3. Production Monitoring

| Signal | Tool | Alert Condition |
|---|---|---|
| Recommendation distribution | Langfuse + dashboard | >20% week-over-week shift in Continue/Review/No-Go share |
| Guardrail override rate | Langfuse custom metric | Spike >3σ above 30-day rolling baseline |
| Rationale length | Langfuse metadata | Consistently <50 words (shortcuts) or >500 words (verbosity/hallucination) |
| Tool call completeness | Langfuse trace | Agent completes <3 tool calls — prompt drift or model regression |
| LLM latency p95 | Langfuse + OpenAI dashboard | >10s p95 — model degradation or traffic spike |
| Input data null rates | DataLoader metrics on startup | Null rate spike in key fields vs. historical baseline |
| Analyst override rate | Application audit log | >30% of agent verdicts overridden in a rolling 7-day window |
| Token cost per assessment | OpenAI usage API | >2× baseline cost per assessment — runaway loop or prompt injection |

### Drift detection

For data drift, compute Population Stability Index (PSI) on the input distribution of the three key ratio metrics (`budget_utilisation_ratio`, `retention_rate`, `business_type_risk_tier`) weekly against the distribution at deployment time. PSI > 0.25 on any metric should trigger a manual data quality review.

For model drift, if the underlying LLM is updated by the provider, run the offline evaluation set immediately and compare F1 scores. A >5% degradation on any class triggers a rollback to the previous model version.

---

## 4. Example Assessments

| CompanyID | Expected path | Guardrail | Notes |
|---|---|---|---|
| `COMPANY_0088` | **Continue** | No | Active; util ≈ 5.9×; retention ≈ 0.47 |
| `COMPANY_0001` | **Review** | No | Active; missing budget utilisation; weak retention |
| `COMPANY_0093` | **No-Go** | Yes — status hard stop | Closed; shows LLM override in UI |

Streamlit demos: sidebar → **Demonstration profiles**.  
CLI: `python scripts/batch_assess.py --demo-three`
