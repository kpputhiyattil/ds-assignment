"""
Credit Assessment Agent

Orchestrates the multi-step tool-calling loop via LangGraph's ReAct agent.
Supports OpenAI (cloud) and any OpenAI-compatible self-hosted LLM
(Ollama, vLLM, LM Studio) via the LLM_PROVIDER env var.

LLM configuration (via .env)
------------------------------
  LLM_PROVIDER=openai            → OpenAI hosted (requires OPENAI_API_KEY)
  LLM_PROVIDER=openai_compatible → Self-hosted (requires LLM_BASE_URL + LLM_MODEL)

See .env.example for full configuration reference.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.guardrails import GuardrailEngine
from agent.models import AssessmentResult, Confidence, Recommendation
from agent.observability import get_langfuse_handler
from agent.prompts import SYSTEM_PROMPT
from agent.tools.business_profile import check_business_profile
from agent.tools.client_engagement import assess_client_engagement
from agent.tools.financial_health import compute_financial_health

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_OPENAI = "gpt-4o-2024-08-06"
DEFAULT_MODEL_COMPATIBLE = "llama3.1:8b"
AGENT_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm() -> ChatOpenAI:
    """
    Construct the LLM client based on LLM_PROVIDER env var.

    LLM_PROVIDER=openai (default)
        Uses OpenAI's hosted API. Requires OPENAI_API_KEY.

    LLM_PROVIDER=openai_compatible
        Uses any server that speaks the OpenAI REST API (Ollama, vLLM, LM Studio).
        Requires LLM_BASE_URL and LLM_MODEL. LLM_API_KEY defaults to "ollama"
        (Ollama ignores it; vLLM may require a real key).

    Temperature is forced to 0 for deterministic, auditable credit decisions.
    Override with LLM_TEMPERATURE env var if needed.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Get your key from https://platform.openai.com/api-keys"
            )
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL_OPENAI)
        logger.info("LLM: OpenAI — model=%s temperature=%s", model, temperature)
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai_compatible":
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        if not base_url:
            raise EnvironmentError(
                "LLM_PROVIDER=openai_compatible but LLM_BASE_URL is not set. "
                "Example: LLM_BASE_URL=http://localhost:11434/v1"
            )
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL_COMPATIBLE)
        api_key = os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores this
        logger.info(
            "LLM: OpenAI-compatible — base_url=%s model=%s temperature=%s",
            base_url,
            model,
            temperature,
        )
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Valid values: 'openai', 'openai_compatible'"
        )


# ---------------------------------------------------------------------------
# JSON verdict parser
# ---------------------------------------------------------------------------

def _parse_verdict_json(content: str) -> dict:
    """
    Extract and parse the JSON verdict from the final AI message content.

    Tries three strategies in order:
    1. Direct json.loads (content is pure JSON)
    2. Extract first {...} block via brace matching (content has surrounding text)
    3. ast.literal_eval fallback (Python-repr-style dict)

    Raises ValueError if none succeed.
    """
    content = content.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first balanced {...} block
    start = content.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(content[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 3: ast fallback
    try:
        result = ast.literal_eval(content)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    raise ValueError(
        f"Could not parse JSON verdict from LLM output. Raw content:\n{content[:500]}"
    )


def _build_assessment_result(
    verdict_dict: dict,
    tool_outputs: dict[str, Any],
    steps: list[BaseMessage],
) -> AssessmentResult:
    """
    Coerce the parsed JSON dict into a typed AssessmentResult.
    Falls back gracefully for unrecognised enum values.
    """
    raw_rec = verdict_dict.get("recommendation", "Review")
    raw_conf = verdict_dict.get("confidence", "low")
    rationale = verdict_dict.get("rationale", "No rationale provided.")

    try:
        recommendation = Recommendation(raw_rec)
    except ValueError:
        logger.warning("Unrecognised recommendation '%s' — defaulting to Review", raw_rec)
        recommendation = Recommendation.REVIEW

    try:
        confidence = Confidence(raw_conf)
    except ValueError:
        logger.warning("Unrecognised confidence '%s' — defaulting to low", raw_conf)
        confidence = Confidence.LOW

    return AssessmentResult(
        recommendation=recommendation,
        confidence=confidence,
        rationale=rationale,
        tool_outputs=tool_outputs,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Tool output extractor
# ---------------------------------------------------------------------------

def _extract_tool_outputs(messages: list[BaseMessage]) -> dict[str, Any]:
    """
    Parse ToolMessage contents from the LangGraph message list.

    Returns a dict keyed by canonical tool name:
        "financial_health", "client_engagement", "business_profile"
    """
    tool_name_map = {
        "compute_financial_health": "financial_health",
        "assess_client_engagement": "client_engagement",
        "check_business_profile": "business_profile",
    }
    outputs: dict[str, Any] = {}

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = getattr(msg, "name", "") or ""
        canonical = tool_name_map.get(tool_name, tool_name)
        content = msg.content

        # Tool content is either a dict (structured) or a JSON/repr string
        if isinstance(content, dict):
            outputs[canonical] = content
        elif isinstance(content, str):
            try:
                outputs[canonical] = json.loads(content)
            except json.JSONDecodeError:
                try:
                    outputs[canonical] = ast.literal_eval(content)
                except Exception:
                    outputs[canonical] = {"raw": content}
        else:
            outputs[canonical] = content

    return outputs


# ---------------------------------------------------------------------------
# Credit Agent
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Raised when the agent cannot produce a valid assessment."""


class CreditAgent:
    """
    LangGraph ReAct agent for credit assessment.

    Instantiation is expensive (LLM client setup) — create once and reuse
    (e.g., via Streamlit's @st.cache_resource).

    Usage
    -----
    agent = CreditAgent()
    result = agent.assess(company_profile_dict)
    """

    def __init__(self) -> None:
        self._llm = build_llm()
        self._tools = [
            compute_financial_health,
            assess_client_engagement,
            check_business_profile,
        ]
        self._graph = create_react_agent(
            self._llm,
            tools=self._tools,
        )
        self._guardrails = GuardrailEngine()
        self._langfuse_handler = get_langfuse_handler()
        logger.info("CreditAgent initialised with %d tools", len(self._tools))

    def assess(self, company_profile: dict) -> AssessmentResult:
        """
        Run a full credit assessment for the given company profile.

        Wraps _run_agent with:
        - Tenacity retry (3 attempts, exponential backoff) for transient errors
        - 60-second wall-clock timeout
        - Catches all exceptions and returns an error AssessmentResult

        Parameters
        ----------
        company_profile : dict returned by DataLoader.get_company()

        Returns
        -------
        AssessmentResult — always returns, never raises.
        """
        company_id = company_profile.get("CompanyID", "unknown")
        logger.info("Starting assessment for company_id=%s", company_id)

        try:
            result = self._assess_with_timeout(company_profile)
        except FuturesTimeoutError:
            msg = f"Assessment timed out after {AGENT_TIMEOUT_SECONDS}s"
            logger.error(msg + " for company_id=%s", company_id)
            return AssessmentResult.error_result(msg)
        except AgentError as exc:
            logger.error("AgentError for company_id=%s: %s", company_id, exc)
            return AssessmentResult.error_result(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error for company_id=%s", company_id)
            return AssessmentResult.error_result(f"Unexpected error: {exc}")

        logger.info(
            "Assessment complete company_id=%s recommendation=%s confidence=%s guardrail=%s",
            company_id,
            result.recommendation.value,
            result.confidence.value,
            result.guardrail_fired,
        )
        return result

    def _assess_with_timeout(self, company_profile: dict) -> AssessmentResult:
        """Run _run_agent_with_retry in a thread with a wall-clock timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_agent_with_retry, company_profile)
            return future.result(timeout=AGENT_TIMEOUT_SECONDS)

    def _run_agent_with_retry(self, company_profile: dict) -> AssessmentResult:
        """Tenacity retry wrapper around _run_agent."""

        @retry(
            retry=retry_if_exception_type((AgentError, Exception)),
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _inner() -> AssessmentResult:
            return self._run_agent(company_profile)

        return _inner()

    def _run_agent(self, company_profile: dict) -> AssessmentResult:
        """
        Core agent invocation.

        1. Build the human message embedding the company profile
        2. Invoke the LangGraph ReAct graph
        3. Extract tool outputs from ToolMessages
        4. Parse the JSON verdict from the final AIMessage
        5. Run the guardrail engine
        6. Return the final AssessmentResult
        """
        human_message = self._build_human_message(company_profile)

        config: dict = {}
        if self._langfuse_handler:
            config["callbacks"] = [self._langfuse_handler]

        # Invoke the LangGraph ReAct agent
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=human_message)]},
            config=config,
        )

        messages: list[BaseMessage] = result.get("messages", [])

        # Extract structured tool outputs
        tool_outputs = _extract_tool_outputs(messages)

        # Find the final AI message containing the verdict JSON
        verdict_content = self._extract_final_content(messages)
        if not verdict_content:
            raise AgentError("Agent produced no final text response.")

        # Parse the JSON verdict
        try:
            verdict_dict = _parse_verdict_json(verdict_content)
        except ValueError as exc:
            raise AgentError(str(exc)) from exc

        # Build typed AssessmentResult
        assessment = _build_assessment_result(verdict_dict, tool_outputs, messages)

        # Run hard guardrails — may override recommendation
        return self._guardrails.run(tool_outputs, assessment)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_human_message(company_profile: dict) -> str:
        """
        Embed the company profile dict into a human message.
        Formats as readable key: value pairs for the LLM.
        """
        profile_lines = "\n".join(
            f"  {k}: {v}" for k, v in company_profile.items()
        )
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"## Company Profile\n\n"
            f"{profile_lines}\n\n"
            f"Please assess this company for growth capital eligibility. "
            f"Call all three tools, then produce your JSON verdict."
        )

    @staticmethod
    def _extract_final_content(messages: list[BaseMessage]) -> Optional[str]:
        """
        Find the last AIMessage with non-empty text content.
        Skips AIMessages that only contain tool_calls (no text yet).
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
                # content can be a list of blocks (e.g. OpenAI tool_call style)
                if isinstance(content, list):
                    text_blocks = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    combined = " ".join(text_blocks).strip()
                    if combined:
                        return combined
        return None
