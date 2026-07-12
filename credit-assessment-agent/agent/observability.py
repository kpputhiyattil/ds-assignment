"""
Langfuse observability setup.

Returns a LangfuseCallbackHandler if credentials are configured, else None.
The agent passes this into config={"callbacks": [handler]} — if None, tracing
is silently skipped and the agent continues without it.

Tracing is intentionally non-blocking: a Langfuse outage must never take
down the assessment pipeline.
"""

from __future__ import annotations

import logging

from agent.config import get_settings

logger = logging.getLogger(__name__)


def get_langfuse_handler():
    """
    Return a configured LangfuseCallbackHandler or None.

    Credentials are read from Settings (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST). Returns None silently if any are missing or the package is
    not installed — tracing is optional, never a hard dependency.
    """
    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.debug(
            "Langfuse keys not configured — tracing disabled. "
            "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable."
        )
        return None

    try:
        # Compatibility shim: langchain 0.2+ removed the `debug` module attribute
        # but some Langfuse versions still try to read/write it.
        import agent.compat  # noqa: F401 — ensure langchain attrs patched before langfuse loads
        from langfuse.callback import CallbackHandler  # type: ignore

        handler = CallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled → %s", settings.langfuse_host)
        return handler

    except ImportError:
        logger.warning(
            "langfuse package not installed — tracing disabled. "
            "Install with: pip install langfuse"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse handler: %s", exc)
        return None
