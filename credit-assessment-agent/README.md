# Credit Assessment Agent

An LLM-powered credit assessment agent that evaluates company applications for growth capital using a ReAct-style agent with deterministic tool calls and hard guardrails.

## Architecture

```
Streamlit UI → DataLoader → LangGraph ReAct Agent → [3 Tools] → Guardrail Engine → AssessmentResult
                                                                        ↓
                                                               Langfuse (traces)
```

The agent calls three deterministic Python tools to compute financial, engagement, and profile signals, then asks GPT-4o to synthesise a structured verdict. Hard guardrails override the LLM if critical conditions are met.

## Setup

### 1. Install dependencies

```bash
# Using uv (recommended)
pip install uv
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip directly
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required)
# Langfuse keys are optional — the agent runs without them
```

### 3. Add data

Place `Companies.parquet` in the project root (or set `DATA_PATH` in `.env`).

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
credit-assessment-agent/
├── Companies.parquet               # Source data (gitignored)
├── pyproject.toml                  # Dependencies
├── .env.example                    # Environment variable template
├── .env                            # Actual secrets (gitignored)
│
├── app/
│   └── streamlit_app.py            # Streamlit UI
│
├── agent/
│   ├── credit_agent.py             # LangGraph ReAct agent
│   ├── data_loader.py              # Parquet loader
│   ├── guardrails.py               # Hard rule engine
│   ├── models.py                   # AssessmentResult dataclass + enums
│   ├── observability.py            # Langfuse setup
│   ├── prompts.py                  # System prompt template
│   └── tools/
│       ├── financial_health.py     # Tool 1: financial metrics
│       ├── client_engagement.py    # Tool 2: engagement metrics
│       └── business_profile.py     # Tool 3: profile risk signals
│
└── tests/
    ├── test_financial_health.py
    ├── test_client_engagement.py
    ├── test_business_profile.py
    ├── test_guardrails.py
    └── test_agent_integration.py
```

## Guardrails

Two hard rules override the LLM verdict regardless of its reasoning:

1. **Company Status Hard Stop** — any non-active status (`inactive`, `suspended`, `closed`) → `No-Go`
2. **Extreme Budget Burn** — revenue covers <10% of monthly budget → `No-Go`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | No | Model name (default: `gpt-4o-2024-08-06`) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key for tracing |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key for tracing |
| `LANGFUSE_HOST` | No | Langfuse host (default: `https://cloud.langfuse.com`) |
| `DATA_PATH` | No | Path to parquet file (default: `./Companies.parquet`) |
