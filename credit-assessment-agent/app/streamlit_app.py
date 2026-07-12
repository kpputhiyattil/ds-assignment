"""
Streamlit UI — LLM-Powered Credit Assessment Agent

Layout
------
Sidebar  : Company selector + Run Assessment button + company profile preview
Main     : Verdict panel (recommendation + confidence + rationale)
           Guardrail warning banner (if triggered)
           Tool Outputs expander (all computed metrics)
           Agent Reasoning Steps expander (LangGraph trace)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# Path setup — must run before importing the local `agent` package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# LangChain compatibility shim — must run before any langchain/langgraph import.
# langchain 0.2+ removed the module-level `debug` attribute that older versions
# of langgraph, langfuse, and langchain_community still reference.
# ---------------------------------------------------------------------------
import agent.compat  # noqa: F401, E402 — patches langchain 0.2+ missing attributes

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Assessment Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Safe imports — deferred so import errors appear in the UI, not as crashes
# ---------------------------------------------------------------------------

def _import_agent_modules():
    """
    Import agent modules lazily so any import-time failures
    (missing packages, bad config) are caught and displayed in the UI.
    """
    from agent.credit_agent import CreditAgent  # noqa: E402
    from agent.data_loader import DataLoader, DataLoaderError  # noqa: E402
    from agent.models import AssessmentResult, Confidence, Recommendation  # noqa: E402
    from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

    return {
        "CreditAgent": CreditAgent,
        "DataLoader": DataLoader,
        "DataLoaderError": DataLoaderError,
        "AssessmentResult": AssessmentResult,
        "Confidence": Confidence,
        "Recommendation": Recommendation,
        "AIMessage": AIMessage,
        "ToolMessage": ToolMessage,
    }


# ---------------------------------------------------------------------------
# User-friendly error display
# ---------------------------------------------------------------------------

def _show_error(title: str, message: str, *, details: str = "", hint: str = "") -> None:
    """Render a structured error block with optional details and actionable hint."""
    st.error(f"**{title}**\n\n{message}")
    logger.error("Error: %s", message)
    if hint:
        st.info(f"**How to fix:** {hint}")
        logger.error("Hint: %s", hint)
    if details:
        with st.expander("Technical details", expanded=False):
            st.code(details, language="text")
            logger.error("Details: %s", details)
    logger.error("Traceback: %s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Cached resources (instantiated once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading company data…")
def get_data_loader(modules: dict):
    DataLoader = modules["DataLoader"]
    DataLoaderError = modules["DataLoaderError"]

    data_path = os.getenv("DATA_PATH", str(ROOT / "Companies.parquet"))
    resolved = Path(data_path)
    if not resolved.is_absolute():
        resolved = ROOT / data_path

    try:
        return DataLoader(str(resolved))
    except DataLoaderError as exc:
        logger.error("DataLoader failed: %s", exc)
        return ("data_error", str(exc))
    except Exception as exc:
        logger.exception("Unexpected error loading data")
        return ("data_error", f"Unexpected error loading data: {exc}")


# Bump this to force @st.cache_resource to rebuild CreditAgent after LLM client fixes.
_AGENT_CACHE_VERSION = "gpt5-max-completion-tokens-v3"


def get_agent(modules: dict, llm_provider: str, llm_model: str, cache_version: str):
    """
    Build a fresh CreditAgent each run.

    Not cached: a stale ChatOpenAI (still sending max_tokens) was surviving
    Streamlit reruns and causing gpt-5 400 errors.
    """
    del llm_provider, llm_model, cache_version  # kept for call-site compatibility
    CreditAgent = modules["CreditAgent"]
    try:
        return CreditAgent()
    except Exception as exc:
        logger.exception("Agent initialisation failed")
        return ("agent_error", str(exc))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _recommendation_ui(recommendation, Recommendation) -> None:
    """Render a colour-coded recommendation block."""
    colour_map = {
        Recommendation.CONTINUE: ("🟢", "success", "Continue"),
        Recommendation.REVIEW: ("🟡", "warning", "Review"),
        Recommendation.NO_GO: ("🔴", "error", "No-Go"),
    }
    entry = colour_map.get(recommendation)
    if entry:
        icon, style, label = entry
        getattr(st, style)(f"## {icon} {label}")
    else:
        st.warning(f"## ⚪ {recommendation.value if hasattr(recommendation, 'value') else recommendation}")


def _confidence_badge(confidence, Confidence) -> str:
    badge_map = {
        Confidence.HIGH: "🔵 High",
        Confidence.MEDIUM: "🟠 Medium",
        Confidence.LOW: "⚪ Low",
    }
    return badge_map.get(confidence, str(confidence.value if hasattr(confidence, "value") else confidence))


def _render_tool_outputs(tool_outputs: dict[str, Any]) -> None:
    """Render all tool outputs as formatted metric tables inside an expander."""
    with st.expander("📊 Tool Outputs — Computed Metrics", expanded=False):
        if not tool_outputs:
            st.info("No tool outputs recorded.")
            return

        for tool_name, outputs in tool_outputs.items():
            st.markdown(f"**{tool_name.replace('_', ' ').title()}**")
            if not isinstance(outputs, dict):
                st.code(str(outputs))
                continue

            issues = outputs.get("data_quality_issues", [])
            rows = []
            for key, value in outputs.items():
                if key == "data_quality_issues":
                    continue
                if value is None:
                    formatted = "—"
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                rows.append({"Metric": key.replace("_", " ").title(), "Value": formatted})

            if rows:
                st.table(rows)

            if issues:
                st.caption(f"⚠️ Data quality issues: {', '.join(issues)}")
            st.divider()


def _render_reasoning_steps(steps: list, modules: dict) -> None:
    """Render the LangGraph message trace inside an expander."""
    AIMessage = modules["AIMessage"]
    ToolMessage = modules["ToolMessage"]

    with st.expander("🔍 Agent Reasoning Steps", expanded=False):
        if not steps:
            st.info("No reasoning steps recorded.")
            return

        tool_step = 0
        for msg in steps:
            if isinstance(msg, ToolMessage):
                tool_step += 1
                tool_name = getattr(msg, "name", "unknown_tool")
                content = msg.content

                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        pass

                st.markdown(f"**Step {tool_step} — Tool call: `{tool_name}`**")
                if isinstance(content, dict):
                    st.json(content)
                else:
                    st.code(str(content))

            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    st.markdown("**Final LLM reasoning (pre-guardrail):**")
                    st.markdown(content)


def _render_company_preview(profile: dict) -> None:
    """Show key company fields in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Company Preview**")
    preview_keys = [
        "CompanyID", "CompanyStatus", "PrimaryType",
        "YearFounded", "OriginCountry", "Rank",
    ]
    for key in preview_keys:
        val = profile.get(key, "—")
        if val is None:
            val = "—"
        st.sidebar.markdown(f"**{key}:** {val}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # Drop any historically cached agent/loader that still had max_tokens wired in.
    if not st.session_state.get("_llm_cache_cleared_v3"):
        st.cache_resource.clear()
        st.session_state["_llm_cache_cleared_v3"] = True

    st.title("🏦 Credit Assessment Agent")
    st.caption(
        "LLM-powered growth capital eligibility assessment. "
        "Powered by a LangGraph ReAct agent with hard guardrails."
    )

    # ------------------------------------------------------------------
    # Import agent modules — display import errors cleanly
    # ------------------------------------------------------------------
    try:
        modules = _import_agent_modules()
    except ImportError as exc:
        _show_error(
            "Missing dependency",
            f"A required package could not be imported: `{exc.name or exc}`",
            details=traceback.format_exc(),
            hint="Run `pip install -e \".[dev]\"` to install all dependencies.",
        )
        st.stop()
    except Exception as exc:
        _show_error(
            "Startup error",
            "Failed to initialise the application.",
            details=traceback.format_exc(),
            hint="Check your `.env` file and installed packages.",
        )
        st.stop()

    Recommendation = modules["Recommendation"]
    Confidence = modules["Confidence"]

    # ------------------------------------------------------------------
    # Load resources — handle data and agent init errors
    # ------------------------------------------------------------------
    loader = get_data_loader(modules)
    if isinstance(loader, tuple) and loader[0] == "data_error":
        error_msg = loader[1]
        if "not found" in error_msg.lower():
            _show_error(
                "Data file not found",
                error_msg,
                hint=(
                    "Place `Companies.parquet` in the project root, or set "
                    "`DATA_PATH` in your `.env` to the correct path. "
                    f"Current DATA_PATH resolves to: `{os.getenv('DATA_PATH', './Companies.parquet')}`"
                ),
            )
        elif "failed to read" in error_msg.lower() or "parquet" in error_msg.lower():
            _show_error(
                "Data file is corrupt or unreadable",
                error_msg,
                hint="Ensure the file is a valid Apache Parquet file. Try re-downloading it.",
            )
        else:
            _show_error("Data loading error", error_msg)
        st.stop()
        return

    # Include provider/model/version in the cache key so LLM client fixes rebuild.
    agent = get_agent(
        modules,
        os.getenv("LLM_PROVIDER", "openai"),
        os.getenv("LLM_MODEL", ""),
        _AGENT_CACHE_VERSION,
    )
    if isinstance(agent, tuple) and agent[0] == "agent_error":
        error_msg = agent[1]
        if "openai_api_key" in error_msg.lower() or "api key" in error_msg.lower():
            _show_error(
                "OpenAI API key not configured",
                error_msg,
                hint=(
                    "Set `OPENAI_API_KEY` in your `.env` file. "
                    "Get a key from https://platform.openai.com/api-keys"
                ),
            )
        elif "llm_base_url" in error_msg.lower() or "base_url" in error_msg.lower():
            _show_error(
                "LLM base URL not configured",
                error_msg,
                hint=(
                    "Set `LLM_BASE_URL` in your `.env` file. "
                    "Example: `LLM_BASE_URL=http://localhost:11434/v1` for Ollama."
                ),
            )
        elif "llm_provider" in error_msg.lower() or "provider" in error_msg.lower():
            _show_error(
                "Invalid LLM provider",
                error_msg,
                hint="Set `LLM_PROVIDER` in your `.env` to either `openai` or `openai_compatible`.",
            )
        else:
            _show_error(
                "Agent initialisation failed",
                error_msg,
                hint="Check your `.env` configuration and ensure all required packages are installed.",
            )
        st.stop()
        return

    # -----------------------------------------------------------------------
    # Sidebar — company selector
    # -----------------------------------------------------------------------
    st.sidebar.header("Select Company")

    try:
        company_ids = loader.list_company_ids()
    except Exception as exc:
        _show_error(
            "Failed to list companies",
            str(exc),
            details=traceback.format_exc(),
            hint="The parquet file may have an unexpected schema. Check that a 'CompanyID' column exists.",
        )
        st.stop()
        return

    if not company_ids:
        _show_error(
            "Empty dataset",
            "No companies found in the dataset.",
            hint="Verify your `Companies.parquet` file is not empty.",
        )
        st.stop()

    selected_id = st.sidebar.selectbox(
        "Company ID",
        options=company_ids,
        help="Select a company to assess for growth capital eligibility.",
    )

    try:
        profile = loader.get_company(selected_id)
    except KeyError:
        _show_error(
            "Company not found",
            f"Company `{selected_id}` does not exist in the dataset.",
            hint="Select a different company from the dropdown.",
        )
        st.stop()
    except Exception as exc:
        _show_error(
            "Failed to load company data",
            f"Could not retrieve data for `{selected_id}`: {exc}",
            details=traceback.format_exc(),
        )
        st.stop()

    _render_company_preview(profile)

    run_button = st.sidebar.button(
        "▶ Run Assessment",
        type="primary",
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Hard guardrails override the LLM if: "
        "the company is non-active, or revenue covers <10% of monthly budget."
    )

    # -----------------------------------------------------------------------
    # Main area — results
    # -----------------------------------------------------------------------
    if not run_button:
        st.info(
            "👈 Select a company from the sidebar and click **Run Assessment** to begin."
        )
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies in dataset", loader.shape[0])
        with col2:
            st.metric("Fields per company", loader.shape[1])
        return

    # -----------------------------------------------------------------------
    # Run assessment with error handling
    # -----------------------------------------------------------------------
    try:
        with st.spinner(f"Assessing {selected_id}… (may take 15–30 seconds)"):
            result = agent.assess(profile)
    except Exception as exc:
        logger.exception("Assessment crashed for %s", selected_id)
        _show_error(
            "Assessment crashed",
            f"An unexpected error occurred while assessing `{selected_id}`.",
            details=traceback.format_exc(),
            hint=(
                "This may be a transient issue. Try again, or check that your "
                "LLM provider is reachable and your API key is valid."
            ),
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Error result from agent (timeout, parse failure, LLM error, etc.)
    # -----------------------------------------------------------------------
    if result.is_error():
        error_msg = result.error or "Unknown error"

        if "timed out" in error_msg.lower():
            _show_error(
                "Assessment timed out",
                f"The assessment for `{selected_id}` did not complete in time.",
                details=error_msg,
                hint="The LLM may be slow or unresponsive. Try again, or check your network connection.",
            )
        elif "api" in error_msg.lower() or "auth" in error_msg.lower() or "401" in error_msg:
            _show_error(
                "LLM API error",
                f"The LLM provider returned an error.",
                details=error_msg,
                hint=(
                    "Check that your API key is valid and has sufficient credits. "
                    "Also verify the model name in LLM_MODEL is correct."
                ),
            )
        elif "parse" in error_msg.lower() or "json" in error_msg.lower():
            _show_error(
                "Failed to parse LLM response",
                "The agent could not extract a structured verdict from the LLM output.",
                details=error_msg,
                hint="This is usually transient. Try running the assessment again.",
            )
        else:
            _show_error(
                "Assessment failed",
                f"Could not complete the assessment for `{selected_id}`.",
                details=error_msg,
                hint="Try running the assessment again. If the problem persists, check the logs.",
            )
        st.stop()

    # -----------------------------------------------------------------------
    # Guardrail banner — shown prominently above the verdict
    # -----------------------------------------------------------------------
    if result.guardrail_fired:
        st.warning(
            f"⚠️ **Guardrail Override Applied**\n\n{result.guardrail_reason}",
            icon="🚨",
        )

    # -----------------------------------------------------------------------
    # Verdict panel
    # -----------------------------------------------------------------------
    st.markdown(f"### Assessment: `{selected_id}`")

    col_rec, col_conf = st.columns([2, 1])

    with col_rec:
        _recommendation_ui(result.recommendation, Recommendation)

    with col_conf:
        st.metric(
            label="Confidence",
            value=_confidence_badge(result.confidence, Confidence),
        )
        if result.guardrail_fired:
            st.caption("🔒 Guardrail override — confidence forced to High")

    st.markdown("#### Rationale")
    st.markdown(result.rationale)

    st.divider()

    # -----------------------------------------------------------------------
    # Tool outputs + reasoning trace
    # -----------------------------------------------------------------------
    _render_tool_outputs(result.tool_outputs)
    _render_reasoning_steps(result.steps, modules)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.caption(
        "⚠️ This assessment is a decision-support tool, not a final credit decision. "
        "Always review with a qualified credit analyst before proceeding."
    )


if __name__ == "__main__":
    main()
