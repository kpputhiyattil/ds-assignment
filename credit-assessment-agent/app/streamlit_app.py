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

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from agent.credit_agent import CreditAgent  # noqa: E402
from agent.data_loader import DataLoader, DataLoaderError  # noqa: E402
from agent.models import AssessmentResult, Confidence, Recommendation  # noqa: E402

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
# Cached resources (instantiated once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading company data…")
def get_data_loader() -> DataLoader | None:
    data_path = os.getenv("DATA_PATH", str(ROOT / "Companies.parquet"))
    try:
        return DataLoader(data_path)
    except DataLoaderError as exc:
        st.error(f"⚠️ Could not load data: {exc}")
        return None


@st.cache_resource(show_spinner="Initialising agent…")
def get_agent() -> CreditAgent | None:
    try:
        return CreditAgent()
    except EnvironmentError as exc:
        st.error(f"⚠️ Agent configuration error: {exc}")
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _recommendation_ui(recommendation: Recommendation) -> None:
    """Render a colour-coded recommendation block."""
    colour_map = {
        Recommendation.CONTINUE: ("🟢", "success", "Continue"),
        Recommendation.REVIEW: ("🟡", "warning", "Review"),
        Recommendation.NO_GO: ("🔴", "error", "No-Go"),
    }
    icon, style, label = colour_map[recommendation]
    getattr(st, style)(f"## {icon} {label}")


def _confidence_badge(confidence: Confidence) -> str:
    badge_map = {
        Confidence.HIGH: "🔵 High",
        Confidence.MEDIUM: "🟠 Medium",
        Confidence.LOW: "⚪ Low",
    }
    return badge_map.get(confidence, str(confidence))


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

            issues = outputs.pop("data_quality_issues", [])
            rows = []
            for key, value in outputs.items():
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

            # Restore issues key for completeness
            outputs["data_quality_issues"] = issues
            st.divider()


def _render_reasoning_steps(steps: list) -> None:
    """Render the LangGraph message trace inside an expander."""
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

                # Try to pretty-print JSON content
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
                # Only show the final text AI message (skip pure tool-call AIMessages)
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
    st.title("🏦 Credit Assessment Agent")
    st.caption(
        "LLM-powered growth capital eligibility assessment. "
        "Powered by a LangGraph ReAct agent with hard guardrails."
    )

    # Load resources
    loader = get_data_loader()
    agent = get_agent()

    if loader is None or agent is None:
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar — company selector
    # -----------------------------------------------------------------------
    st.sidebar.header("Select Company")

    company_ids = loader.list_company_ids()
    if not company_ids:
        st.error("No companies found in the dataset.")
        st.stop()

    selected_id = st.sidebar.selectbox(
        "Company ID",
        options=company_ids,
        help="Select a company to assess for growth capital eligibility.",
    )

    # Load company profile
    try:
        profile = loader.get_company(selected_id)
    except Exception as exc:
        st.error(f"Could not load company '{selected_id}': {exc}")
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

        # Show dataset summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies in dataset", loader.shape[0])
        with col2:
            st.metric("Fields per company", loader.shape[1])
        return

    # Run the assessment
    with st.spinner(f"Assessing {selected_id}… (may take 15–30 seconds)"):
        result: AssessmentResult = agent.assess(profile)

    # -----------------------------------------------------------------------
    # Error state
    # -----------------------------------------------------------------------
    if result.is_error():
        st.error(f"❌ Assessment failed: {result.error}")
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
        _recommendation_ui(result.recommendation)

    with col_conf:
        st.metric(
            label="Confidence",
            value=_confidence_badge(result.confidence),
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
    _render_reasoning_steps(result.steps)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.caption(
        "⚠️ This assessment is a decision-support tool, not a final credit decision. "
        "Always review with a qualified credit analyst before proceeding."
    )


if __name__ == "__main__":
    main()
