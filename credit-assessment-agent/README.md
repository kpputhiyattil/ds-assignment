# Credit Assessment Agent

An LLM-powered credit assessment agent that evaluates company applications for
growth capital using a ReAct-style agent with deterministic tool calls and hard
guardrails. Verdict: **Continue / Review / No-Go**, plus confidence and a
natural-language rationale.

Design notes: [`SOLUTION_BLUEPRINT.md`](./SOLUTION_BLUEPRINT.md) ·
Mini report: [`MINI_REPORT.md`](./MINI_REPORT.md).

## Problem framing

| | |
|---|---|
| **Task** | First-pass credit triage from a structured company profile (`Companies.parquet`). |
| **Signals** | Three deterministic tools: financial health, client engagement, business profile. |
| **Guardrails** | Hard overrides on non-active status and extreme budget burn (`util < 10%`). |
| **Delivery** | Streamlit analyst UI + CLI batch assessment + EDA report. |

## Tech stack

Python 3.12 · pandas / pyarrow · LangGraph ReAct · langchain-openai · Streamlit ·
Langfuse (optional) · pydantic-settings.

## Project layout

```
configs via .env               # LLM provider, DATA_PATH, BATCH_SIZE
agent/                         # DataLoader, tools, guardrails, ReAct agent, EDA
app/streamlit_app.py           # Analyst UI
scripts/run_eda.py             # EDA CLI                         (FIRST)
scripts/batch_assess.py        # Batch / demo assessments
reports/eda/                   # Generated EDA (JSON + Markdown)
outputs/                       # Batch JSON + ID lists
data/{raw,processed,artifacts} # raw = input; artifacts = generated
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -e ".[dev]"
# or: pip install -r requirements-dev.txt

Copy-Item .env.example .env   # Windows
# cp .env.example .env        # macOS / Linux
# Edit .env — set OPENAI_API_KEY (or LLM_BASE_URL) and DATA_PATH
```

Requires **Python ≥ 3.12**. Always use the **venv** interpreter for streamlit
(not a system-wide `python.exe`), so dependencies resolve correctly.

Default data path: `DATA_PATH=./data/raw/Companies.parquet`.

## Sharing this project with someone else

### What to send

| Item | Required? | Notes |
|---|---|---|
| Project code (repo / zip) | Yes | Source, tests, reports, `MINI_REPORT.md` |
| `data/raw/Companies.parquet` | Yes | **Gitignored** — must share separately |
| `reports/eda/` | Optional | Skip if they will re-run `make eda` |
| `outputs/` batch JSON | Optional | Useful demos; regenerate with batch script |

### Full rebuild (from raw companies)

Put the parquet at `data/raw/Companies.parquet`, activate the venv, then either:

```bash
make eda
make run
```

or run each stage (Windows-friendly; same order):

```bash
# 1) EDA BEFORE agent / batch (design + guardrail evidence)
python scripts/run_eda.py

# 2) Batch assessment (size from .env BATCH_SIZE)
python scripts/batch_assess.py
# or demos:
python scripts/batch_assess.py --demo-three

# 3) Interactive UI
streamlit run app/streamlit_app.py
```

**Order note:** `run_eda` belongs **before** batch / Streamlit. It documents
missingness, status mixes, and budget-utilisation cutoffs that justify the
10% burn guardrail. Batch assessment needs a live LLM (`OPENAI_API_KEY` or
compatible endpoint).

### Faster path (EDA already included)

If `reports/eda/eda_report.md` is already present:

```bash
pip install -e ".[dev]"
# configure .env
streamlit run app/streamlit_app.py
```

### Sanity checks

| Check | Expected |
|---|---|
| EDA | `reports/eda/eda_report.md` + `eda_report.json` |
| UI | `http://localhost:8501` — select company → Run Assessment |
| Demo CLI | `python scripts/batch_assess.py --demo-three` → 3 verdicts |
| Guardrail | Assess `COMPANY_0093` → banner / `guardrail_fired` |
| Batch | `outputs/batch_*_*.json` + `_ids/` folder |

## Pipeline run order

Stages are built incrementally. Once set up, the full run is:

```bash
make eda          # FIRST — company EDA (missingness, status, util cuts) -> reports/eda/
python scripts/batch_assess.py          # uses BATCH_SIZE from .env
python scripts/batch_assess.py --demo-three
make run          # Streamlit UI
```

**Order note:** `make eda` belongs **before** assessment (design / data
understanding / guardrail threshold evidence). Batch size is configured in
`.env`:

```bash
BATCH_SIZE=150    # or 50, or full
```

| `BATCH_SIZE` | Behaviour |
|---|---|
| `50` | First 50 companies (after `--offset`) |
| `150` | First 150 companies |
| `full` / `all` | Entire dataset |

CLI `--limit` overrides `.env` for a one-off run.

## How to use the Streamlit UI

```bash
streamlit run app/streamlit_app.py
# or: make run
```

1. Open the URL Streamlit prints (usually `http://localhost:8501`).
2. Sidebar → pick a `Company ID`.
3. Click **▶ Run Assessment** (~15–30s).
4. Inspect recommendation, confidence, rationale, tool outputs, and any
   guardrail banner.

**Suggested demo companies:**

| CompanyID | Typical path | Guardrail |
|---|---|---|
| `COMPANY_0088` | Continue | No |
| `COMPANY_0015` | Review | No |
| `COMPANY_0093` | No-Go | Yes — company status hard stop |
| `COMPANY_0002` | No-Go | Yes — extreme budget burn |

## Guardrails

1. **Company Status Hard Stop** — `inactive` / `suspended` / `closed` → `No-Go`
2. **Extreme Budget Burn** — revenue covers <10% of monthly budget → `No-Go`

Threshold rationale is regenerated by EDA (`reports/eda/`). Narrative answers
for the assignment live in `MINI_REPORT.md`.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `openai` or `openai_compatible` |
| `OPENAI_API_KEY` | When provider is `openai` | OpenAI API key |
| `LLM_BASE_URL` | When provider is `openai_compatible` | Base URL (e.g. Ollama `/v1`) |
| `LLM_API_KEY` | No | Key for compatible providers (default `ollama`) |
| `LLM_MODEL` | No | Default `gpt-4o-2024-08-06` |
| `LLM_TEMPERATURE` | No | Default `0.0` |
| `LLM_MAX_TOKENS` | No | Default `8192` |
| `DATA_PATH` | No | Prefer `./data/raw/Companies.parquet` |
| `BATCH_SIZE` | No | `full`, `50`, `150`, or any positive integer (default `150`) |
| `LANGFUSE_*` | No | Optional tracing |

## Build progress

- [x] **Data loader** — parquet load, schema rename, company lookup
- [x] **EDA** — `scripts/run_eda.py` → `reports/eda/`
- [x] **Tools** — financial health, client engagement, business profile
- [x] **Guardrails** — status hard stop + extreme budget burn
- [x] **Agent** — LangGraph ReAct + Continue calibration
- [x] **UI** — Streamlit assess flow
- [x] **Batch** — CLI with `BATCH_SIZE` / `--demo-three`
- [x] **Mini report** — guardrails, evaluation, monitoring, improvements
