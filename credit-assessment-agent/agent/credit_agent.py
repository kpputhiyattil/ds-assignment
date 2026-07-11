"""
Credit Assessment Agent

Orchestrates the multi-step tool-calling loop via LangGraph's ReAct agent.
Supports OpenAI (cloud) and any OpenAI-compatible self-hosted LLM
(Ollama, vLLM, LM Studio) — configured via agent/config.py.

All LLM configuration comes from Settings (agent/config.py).
No raw os.getenv() calls here.
"""

from __future__ import annotations

import ast
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Optional

# ---------------------------------------------------------------------------
# LangChain compatibility shim
# langchain 0.2+ removed the module-level `debug` attribute that some
# versions of langgraph / langfuse / langchain_community still reference.
# Patching it here before any langchain imports prevents AttributeError.
# ---------------------------------------------------------------------------
import agent.compat  # noqa: F401 — patches langchain 0.2+ missing attributes

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent.config import Settings, get_settings
from agent.guardrails import GuardrailEngine
from agent.models import AssessmentResult, Confidence, Recommendation
from agent.observability import get_langfuse_handler
from agent.prompts import SYSTEM_PROMPT
from agent.tools.business_profile import check_business_profile
from agent.tools.client_engagement import assess_client_engagement
from agent.tools.financial_health import compute_financial_health

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _requires_max_completion_tokens(model: str) -> bool:
    """
    Return True for models that reject max_tokens in favour of max_completion_tokens.

    Covers o-series (o1/o3/o4…) and gpt-5*. Strips an optional provider prefix
    (e.g. 'openai/gpt-5') so both bare and routed names match.
    """
    name = model.lower().strip().split("/")[-1]
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


def _is_verdict_dict(value: Any) -> bool:
    """True if value looks like the expected credit verdict JSON object."""
    return isinstance(value, dict) and "recommendation" in value


def _looks_like_tool_call_payload(value: Any) -> bool:
    """True if value is a tool-call list/object, not a credit verdict."""
    if isinstance(value, list) and value:
        first = value[0]
        return isinstance(first, dict) and "name" in first and (
            "arguments" in first or "args" in first
        )
    if isinstance(value, dict):
        return "name" in value and ("arguments" in value or "args" in value) and (
            "recommendation" not in value
        )
    return False


def _promote_text_tool_calls(message: BaseMessage) -> BaseMessage:
    """
    Some OpenAI-compatible servers (e.g. self-hosted adept3o) emit tool calls as
    a JSON array in message.content instead of the native tool_calls field.
    Promote that text into AIMessage.tool_calls so LangGraph's ReAct loop runs.
    """
    if not isinstance(message, AIMessage):
        return message
    if getattr(message, "tool_calls", None):
        return message

    content = message.content
    if not isinstance(content, str) or not content.strip():
        return message

    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError:
        return message

    if not _looks_like_tool_call_payload(parsed):
        return message

    calls = parsed if isinstance(parsed, list) else [parsed]
    tool_calls: list[dict[str, Any]] = []
    for i, call in enumerate(calls):
        raw_args = call.get("arguments", call.get("args", {}))
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {"raw": raw_args}
        if not isinstance(raw_args, dict):
            raw_args = {"value": raw_args}
        # Many OpenAI-compatible servers require alphanumeric IDs with len >= 9
        call_id = str(call.get("id") or "")
        if len(call_id) < 9 or not call_id.isalnum():
            call_id = f"call{uuid.uuid4().hex[:12]}"
        tool_calls.append(
            {
                "name": call["name"],
                "args": raw_args,
                "id": call_id,
                "type": "tool_call",
            }
        )

    logger.info(
        "Promoted text tool-call payload → native tool_calls: %s",
        [c["name"] for c in tool_calls],
    )
    return AIMessage(
        content="",
        tool_calls=tool_calls,
        additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
        response_metadata=dict(getattr(message, "response_metadata", {}) or {}),
        id=getattr(message, "id", None),
        usage_metadata=getattr(message, "usage_metadata", None),
    )


class CompatibleChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI hardened for gpt-5 param quirks and openai-compatible tool quirks.

    - Remaps max_tokens → max_completion_tokens for gpt-5 / o-series
    - Promotes text-emitted tool-call JSON into native tool_calls (self-hosted)
    """

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        model = str(params.get("model") or "")
        if not _requires_max_completion_tokens(model):
            return params

        if "max_tokens" in params:
            params.setdefault("max_completion_tokens", params.pop("max_tokens"))
            params.pop("max_tokens", None)

        # These models only accept the default temperature (1)
        params["temperature"] = 1
        return params

    @property
    def _default_params(self) -> dict[str, Any]:
        return self._sanitize_params(dict(super()._default_params))

    def _get_request_payload(self, input_, *, stop=None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return self._sanitize_params(payload)

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info=generation_info)
        for generation in result.generations:
            generation.message = _promote_text_tool_calls(generation.message)
        return result


def build_llm(settings: Optional[Settings] = None) -> ChatOpenAI:
    """
    Construct the LLM client from Settings.

    Settings.llm_provider controls which backend is used:
    - 'openai'            → OpenAI hosted API (OPENAI_API_KEY required)
    - 'openai_compatible' → Any OpenAI-compatible server (LLM_BASE_URL required)
                            Works with Ollama, vLLM, LM Studio, Groq, Together AI, etc.

    Validation of required credentials is done in Settings at startup, so
    this function only needs to build the client object.
    """
    cfg = settings or get_settings()

    # langchain-openai 0.1.x only knows the `max_tokens` field. For gpt-5 / o-series
    # the API rejects max_tokens, so the limit must go via model_kwargs as
    # max_completion_tokens. Those models also only accept temperature=1
    # (ChatOpenAI always sends temperature; omitting it defaults to 0.7 and fails).
    if _requires_max_completion_tokens(cfg.llm_model):
        # gpt-5 / o-series spend a large share of the budget on hidden
        # reasoning tokens; 1024 is often entirely consumed before any
        # visible JSON verdict is produced. Floor at 8192 unless raised.
        completion_budget = max(cfg.llm_max_tokens, 8192)
        common_kwargs: dict[str, Any] = {
            "model": cfg.llm_model,
            "temperature": 1,
            "model_kwargs": {"max_completion_tokens": completion_budget},
        }
        temp_log = 1
    else:
        common_kwargs = {
            "model": cfg.llm_model,
            "temperature": cfg.llm_temperature,
            "max_tokens": cfg.llm_max_tokens,
        }
        temp_log = cfg.llm_temperature

    if cfg.is_openai:
        logger.info(
            "LLM: OpenAI — model=%s temperature=%s max_completion_tokens=%s",
            cfg.llm_model,
            temp_log,
            common_kwargs.get("model_kwargs", {}).get("max_completion_tokens")
            or common_kwargs.get("max_tokens"),
        )
        return CompatibleChatOpenAI(api_key=cfg.openai_api_key, **common_kwargs)

    # openai_compatible
    logger.info(
        "LLM: OpenAI-compatible — base_url=%s model=%s temperature=%s",
        cfg.llm_base_url,
        cfg.llm_model,
        temp_log,
    )
    return CompatibleChatOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        **common_kwargs,
    )


# ---------------------------------------------------------------------------
# JSON verdict parser
# ---------------------------------------------------------------------------

def _parse_verdict_json(content: str) -> dict:
    """
    Extract and parse the JSON verdict from the final AI message content.

    Three strategies in priority order:
    1. Direct json.loads  — content is pure JSON
    2. Brace extraction   — content has surrounding prose, find the {...} block
    3. ast.literal_eval   — Python-repr-style dict (rare fallback)

    Raises ValueError if none succeed or the payload is not a verdict dict
    (e.g. a tool-call list accidentally treated as the final answer).
    """
    content = content.strip()
    candidates: list[Any] = []

    try:
        candidates.append(json.loads(content))
    except json.JSONDecodeError:
        pass

    # Find every balanced {...} block (prefer the one that looks like a verdict)
    start = 0
    while True:
        start = content.find("{", start)
        if start == -1:
            break
        depth = 0
        for i, ch in enumerate(content[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidates.append(json.loads(content[start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = i + 1
                    break
        else:
            break

    try:
        candidates.append(ast.literal_eval(content))
    except (ValueError, SyntaxError):
        pass

    for candidate in candidates:
        if _is_verdict_dict(candidate):
            return candidate

    if any(_looks_like_tool_call_payload(c) for c in candidates):
        raise ValueError(
            "Model returned tool-call JSON instead of a verdict. "
            "Expected an object with keys recommendation / confidence / rationale."
        )

    raise ValueError(
        f"Could not parse JSON verdict from LLM output.\nRaw content:\n{content[:500]}"
    )


def _coerce_assessment(
    verdict_dict: dict,
    tool_outputs: dict[str, Any],
    steps: list[BaseMessage],
) -> AssessmentResult:
    """Coerce the parsed JSON dict into a typed AssessmentResult."""
    if not isinstance(verdict_dict, dict):
        raise AgentError(
            f"Expected verdict dict, got {type(verdict_dict).__name__}: {verdict_dict!r:.200}"
        )

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
    _name_map = {
        "compute_financial_health": "financial_health",
        "assess_client_engagement": "client_engagement",
        "check_business_profile": "business_profile",
    }
    outputs: dict[str, Any] = {}

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "") or ""
        canonical = _name_map.get(tool_name, tool_name)
        content = msg.content

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


def _extract_final_ai_content(messages: list[BaseMessage]) -> Optional[str]:
    """
    Find the last AIMessage with non-empty text content that looks like a verdict.

    Skips AIMessages that still contain tool_calls (intermediate ReAct steps)
    and content that is tool-call JSON rather than the final verdict object.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue

        # Intermediate ReAct steps: model is requesting tools, not giving a verdict
        if getattr(msg, "tool_calls", None):
            continue

        content = msg.content
        text: Optional[str] = None

        if isinstance(content, str) and content.strip():
            text = content.strip()
        elif isinstance(content, list):
            # OpenAI structured content blocks (list of dicts with type=text)
            text = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip() or None

        if not text:
            continue

        # Reject tool-call shaped payloads so we keep searching / fail clearly
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and _looks_like_tool_call_payload(parsed):
            continue
        if parsed is not None and isinstance(parsed, dict) and not _is_verdict_dict(parsed):
            # Might still contain a verdict object inside prose — let the parser try
            pass

        return text

    return None


# ---------------------------------------------------------------------------
# Credit Agent
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Raised when the agent cannot produce a valid assessment."""


class CreditAgent:
    """
    LangGraph ReAct agent for credit assessment.

    Instantiation is expensive (LLM client + graph compilation).
    Create once and reuse — in Streamlit via @st.cache_resource.

    Usage
    -----
    agent = CreditAgent()
    result = agent.assess(company_profile_dict)  # always returns, never raises
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or get_settings()
        self._llm = build_llm(self._cfg)
        self._tools = [
            compute_financial_health,
            assess_client_engagement,
            check_business_profile,
        ]
        self._graph = create_react_agent(self._llm, tools=self._tools)
        self._guardrails = GuardrailEngine()
        self._langfuse_handler = get_langfuse_handler()
        logger.info(
            "CreditAgent ready — provider=%s model=%s tools=%d",
            self._cfg.llm_provider,
            self._cfg.llm_model,
            len(self._tools),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, company_profile: dict) -> AssessmentResult:
        """
        Run a full credit assessment for the given company profile.

        Always returns an AssessmentResult — never raises. Errors are
        captured in AssessmentResult.error and surfaced in the UI.

        Wraps _run_agent with:
        - Wall-clock timeout (Settings.agent_timeout_seconds)
        - Tenacity retry (Settings.agent_max_retries, exponential backoff)
        """
        company_id = company_profile.get("CompanyID", "unknown")
        logger.info("Starting assessment — company_id=%s", company_id)

        try:
            result = self._run_with_timeout(company_profile)
        except FuturesTimeoutError:
            msg = f"Assessment timed out after {self._cfg.agent_timeout_seconds}s"
            logger.error("%s — company_id=%s", msg, company_id)
            return AssessmentResult.error_result(msg)
        except AgentError as exc:
            logger.error("AgentError — company_id=%s: %s", company_id, exc)
            return AssessmentResult.error_result(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error — company_id=%s", company_id)
            return AssessmentResult.error_result(f"Unexpected error: {exc}")

        logger.info(
            "Assessment complete — company_id=%s recommendation=%s "
            "confidence=%s guardrail=%s",
            company_id,
            result.recommendation.value,
            result.confidence.value,
            result.guardrail_fired,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: timeout + retry wrappers
    # ------------------------------------------------------------------

    def _run_with_timeout(self, company_profile: dict) -> AssessmentResult:
        """Execute _run_with_retry inside a thread with a wall-clock timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_with_retry, company_profile)
            return future.result(timeout=self._cfg.agent_timeout_seconds)

    def _run_with_retry(self, company_profile: dict) -> AssessmentResult:
        """Tenacity retry wrapper — retries on any exception except AgentError."""

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._cfg.agent_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _inner() -> AssessmentResult:
            return self._run_agent(company_profile)

        return _inner()

    # ------------------------------------------------------------------
    # Internal: core agent invocation
    # ------------------------------------------------------------------

    def _run_agent(self, company_profile: dict) -> AssessmentResult:
        """
        Core LangGraph invocation.

        Steps:
        1. Build human message with embedded company profile
        2. Invoke the ReAct graph
        3. Extract ToolMessage outputs → tool_outputs dict
        4. Parse JSON verdict from final AIMessage
        5. If the model re-emitted tool-call JSON instead of a verdict
           (common with some openai-compatible servers), force a no-tools
           follow-up that asks only for the verdict JSON
        6. Run GuardrailEngine → may override recommendation
        7. Return AssessmentResult
        """
        human_message = self._build_human_message(company_profile)

        invoke_config: dict = {"recursion_limit": 25}
        if self._langfuse_handler:
            invoke_config["callbacks"] = [self._langfuse_handler]

        graph_result = self._graph.invoke(
            {"messages": [HumanMessage(content=human_message)]},
            config=invoke_config,
        )

        messages: list[BaseMessage] = graph_result.get("messages", [])
        tool_outputs = _extract_tool_outputs(messages)

        verdict_dict: Optional[dict] = None
        verdict_text = _extract_final_ai_content(messages)
        if verdict_text:
            try:
                verdict_dict = _parse_verdict_json(verdict_text)
            except ValueError as exc:
                logger.warning(
                    "Primary verdict parse failed (%s) — trying fallback",
                    exc,
                )

        if verdict_dict is None:
            if not tool_outputs:
                raise AgentError(
                    "Agent produced no tool outputs and no JSON verdict. "
                    "The model may not support tool calling correctly."
                )
            logger.info(
                "Requesting verdict fallback from %d tool output(s)",
                len(tool_outputs),
            )
            verdict_dict = self._request_verdict_fallback(tool_outputs, invoke_config)
            # Keep a synthetic AIMessage in the trace for the UI steps panel
            messages = list(messages) + [
                AIMessage(content=json.dumps(verdict_dict))
            ]

        logger.info(
            "Verdict parsed — recommendation=%s confidence=%s",
            verdict_dict.get("recommendation"),
            verdict_dict.get("confidence"),
        )
        assessment = _coerce_assessment(verdict_dict, tool_outputs, messages)
        return self._guardrails.run(tool_outputs, assessment)

    def _request_verdict_fallback(
        self,
        tool_outputs: dict[str, Any],
        invoke_config: dict,
    ) -> dict:
        """
        Ask the bare LLM (no tools bound) for the JSON verdict given tool results.

        Used when a self-hosted / openai-compatible model finishes the ReAct loop
        by re-emitting tool-call JSON instead of recommendation/confidence/rationale.
        """
        prompt = (
            "You are a senior credit analyst. The analysis tools have ALREADY been "
            "run. Do NOT call any tools. Do NOT return a tool-call JSON array. "
            "Using ONLY the tool outputs below, return a single JSON object with "
            "EXACTLY these keys:\n"
            '  "recommendation": "Continue" | "Review" | "No-Go",\n'
            '  "confidence": "low" | "medium" | "high",\n'
            '  "rationale": "<100-150 words citing specific tool metrics>"\n\n'
            "Output ONLY the JSON object — no markdown fences, no preamble, "
            "no tool calls.\n\n"
            f"## Tool outputs\n\n{json.dumps(tool_outputs, indent=2, default=str)}"
        )

        last_error: Optional[Exception] = None
        for attempt in range(2):
            # Unbound LLM — no tools available, so the model must answer in text
            response = self._llm.invoke(
                [HumanMessage(content=prompt)],
                config=invoke_config,
            )
            content = self._message_text(response)

            # Promotion may have turned tool-call text into empty content + tool_calls
            if getattr(response, "tool_calls", None) and not content:
                last_error = AgentError(
                    "Model returned tool calls again during verdict fallback."
                )
                prompt = (
                    "STOP. Do not call tools. Reply with ONLY this JSON shape:\n"
                    '{"recommendation":"Review","confidence":"low","rationale":"..."}\n\n'
                    f"Tool outputs:\n{json.dumps(tool_outputs, default=str)}"
                )
                continue

            if not content:
                last_error = AgentError(
                    "Verdict fallback returned empty content after tools completed."
                )
                continue

            try:
                return _parse_verdict_json(content)
            except ValueError as exc:
                last_error = exc
                prompt = (
                    "Your previous reply was invalid. Return ONLY a JSON object with "
                    "keys recommendation, confidence, rationale. No tool calls.\n\n"
                    f"Tool outputs:\n{json.dumps(tool_outputs, default=str)}"
                )

        raise AgentError(
            f"Verdict fallback failed after tools completed: {last_error}"
        )

    @staticmethod
    def _message_text(message: Any) -> str:
        """Extract plain text from an AIMessage / content blocks."""
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content.strip() if isinstance(content, str) else ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_human_message(company_profile: dict) -> str:
        """Embed the company profile as readable key: value pairs."""
        profile_lines = "\n".join(
            f"  {k}: {v}" for k, v in company_profile.items()
        )
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"## Company Profile\n\n"
            f"{profile_lines}\n\n"
            "Please assess this company for growth capital eligibility. "
            "Call all three tools, then produce your JSON verdict."
        )
