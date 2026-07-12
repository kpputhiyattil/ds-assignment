# Credit Assessment Agent

An LLM-powered credit assessment agent that evaluates company applications for growth capital using a ReAct-style agent with deterministic tool calls and hard guardrails.

## Architecture

```
Streamlit UI / CLI → DataLoader → LangGraph ReAct Agent → [3 Tools] → Guardrail Engine → AssessmentResult
                                                                        ↓
                                                               Langfuse (traces)
```

The agent calls three deterministic Python tools to compute financial, engagement, and profile signals, then asks the LLM to synthesise a structured verdict. Hard guardrails override the LLM if critical conditions are met.

---

## How to run the entire flow

Follow these steps from the project root (`credit-assessment-agent/`).

### 1. Prerequisites

- Python **3.12+**
- An OpenAI API key (or a local OpenAI-compatible server such as Ollama)
- `Companies.parquet` under `data/raw/` (already present in this repo layout)

### 2. Create a virtual environment and install dependencies

```bash
# Using uv (recommended)
pip install uv
uv venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

uv pip install -e ".[dev]"
# Or:
# pip install -r requirements-dev.txt
```

Using Make (if available):

```bash
make install-dev
```

### 3. Configure environment

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
# cp .env.example .env
```

Edit `.env` and set at least:

| Variable | Required | Example |
|---|---|---|
| `LLM_PROVIDER` | Yes | `openai` or `openai_compatible` |
| `OPENAI_API_KEY` | If using OpenAI | `sk-...` |
| `LLM_BASE_URL` | If using compatible provider | `http://localhost:11434/v1` |
| `LLM_MODEL` | No | `gpt-4o-2024-08-06` |
| `DATA_PATH` | No | `./data/raw/Companies.parquet` |
| `LANGFUSE_*` | No | Optional tracing |

Default data path in `.env.example` points at `./data/raw/Companies.parquet`.

### 4. Verify the setup (tests)

```bash
pytest tests/ -v
# or: make test
```

Unit tests cover the three tools, guardrails, and agent integration. They do not require a live LLM for most cases.

### 5. Run the interactive end-to-end flow (Streamlit)

```bash
streamlit run app/streamlit_app.py
# or: make run
```

Then in the browser:

1. Open the URL Streamlit prints (usually `http://localhost:8501`).
2. In the **sidebar**, pick a `Company ID` from the dropdown.
3. Review the **Company Preview** fields.
4. Click **▶ Run Assessment**.
5. Wait ~15–30 seconds while the agent:
   - loads the company row from parquet
   - calls the three tools (financial health, client engagement, business profile)
   - asks the LLM for a structured verdict
   - applies hard guardrails
6. Inspect the main panel:
   - **Recommendation** — Continue / Review / No-Go
   - **Confidence** — low / medium / high
   - **Rationale** — natural-language explanation
   - **Guardrail banner** — shown only if a hard rule overrode the LLM
   - **Tool Outputs** expander — deterministic metrics
   - **Agent Reasoning Steps** expander — LangGraph / tool-call trace

**Suggested demo companies** (pick these IDs in the sidebar):

| CompanyID | Typical path | Guardrail |
|---|---|---|
| `COMPANY_0088` | Continue | No |
| `COMPANY_0001` | Review | No |
| `COMPANY_0093` | No-Go | Yes — company status hard stop |

### 6. Run the same flow from the CLI (batch / demos)

Configure how many companies to assess in `.env` (used when `--limit` is omitted):

```bash
# .env
BATCH_SIZE=150    # or 50, or full (every company)
```

| `BATCH_SIZE` | Behaviour |
|---|---|
| `50` | First 50 companies (after `--offset`) |
| `150` | First 150 companies |
| `full` / `all` | Entire dataset |

Then run:

```bash
# Uses BATCH_SIZE from .env
python scripts/batch_assess.py

# One-off override (does not change .env)
python scripts/batch_assess.py --limit 50
python scripts/batch_assess.py --limit 150 --offset 0
```

Quick three-company demo (Continue / Review / Guardrail No-Go):

```bash
python scripts/batch_assess.py --demo-three
```

Assess a custom list:

```bash
python scripts/batch_assess.py --ids COMPANY_0088,COMPANY_0001,COMPANY_0093
```

Optional explicit output path:

```bash
python scripts/batch_assess.py --out outputs/batch_report.json
```

Results are written under `outputs/`:

- `batch_<BATCH_SIZE>_<timestamp>.json` — full structured results
- `batch_<BATCH_SIZE>_<timestamp>_ids/` — `Continue.txt`, `Review.txt`, `No-Go.txt`, `SUMMARY.txt`

### 7. End-to-end checklist

| Step | Command / action | Success signal |
|---|---|---|
| Install | `uv pip install -e ".[dev]"` | No errors |
| Configure | `.env` with API key + `DATA_PATH` | Settings load without validation errors |
| Tests | `pytest tests/ -v` | All tests pass |
| UI flow | `streamlit run app/streamlit_app.py` → select company → Run | Verdict + tool outputs appear |
| Demo CLI | `python scripts/batch_assess.py --demo-three` | Three verdicts printed; JSON under `outputs/` |
| Guardrail | Assess `COMPANY_0093` | Red / warning banner; `guardrail_fired` |

---

## Project structure

```
credit-assessment-agent/
├── .env.example                    # Environment variable template
├── .env                            # Actual secrets (gitignored)
├── pyproject.toml                  # Dependencies
├── Makefile                        # install / test / run shortcuts
├── MINI_REPORT.md                  # Design notes & demo IDs
│
├── data/raw/
│   └── Companies.parquet           # Source data
│
├── app/
│   └── streamlit_app.py            # Streamlit UI
│
├── agent/
│   ├── credit_agent.py             # LangGraph ReAct agent
│   ├── data_loader.py              # Parquet loader
│   ├── guardrails.py               # Hard rule engine
│   ├── models.py                   # AssessmentResult + enums
│   ├── observability.py            # Langfuse setup
│   ├── prompts.py                  # System prompt
│   ├── config.py                   # Settings from .env
│   └── tools/
│       ├── financial_health.py
│       ├── client_engagement.py
│       └── business_profile.py
│
├── scripts/
│   └── batch_assess.py             # CLI batch / demo assessments
│
├── outputs/                        # Batch JSON + ID lists
└── tests/
```

## Guardrails

Two hard rules override the LLM verdict regardless of its reasoning:

1. **Company Status Hard Stop** — any non-active status (`inactive`, `suspended`, `closed`) → `No-Go`
2. **Extreme Budget Burn** — revenue covers <10% of monthly budget → `No-Go`

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `openai` or `openai_compatible` |
| `OPENAI_API_KEY` | When provider is `openai` | OpenAI API key |
| `LLM_BASE_URL` | When provider is `openai_compatible` | Base URL (e.g. Ollama `/v1`) |
| `LLM_API_KEY` | No | Key for compatible providers (default `ollama`) |
| `LLM_MODEL` | No | Model name (default: `gpt-4o-2024-08-06`) |
| `LLM_TEMPERATURE` | No | Default `0.0` |
| `LLM_MAX_TOKENS` | No | Default `8192` |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key for tracing |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key for tracing |
| `LANGFUSE_HOST` | No | Default `https://cloud.langfuse.com` |
| `DATA_PATH` | No | Path to parquet (default `./Companies.parquet`; prefer `./data/raw/Companies.parquet`) |
| `BATCH_SIZE` | No | Batch report size: `full`, `50`, `150`, or any positive integer (default `150`) |
