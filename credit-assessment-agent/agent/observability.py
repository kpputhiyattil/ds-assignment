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
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_langfuse_handler():
    """
    Return a configured LangfuseCallbackHandler or None.

    Requires all three env vars to be set:
        LANGFUSE_SECRET_KEY
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_HOST  (defaults to https://cloud.langfuse.com)

    Returns None (without raising) if:
    - Keys are missing / empty
    - langfuse package is not installed
    - Langfuse host is unreachable (deferred to first trace attempt)
    """
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

    if not secret_key or not public_key:
        logger.debug(
            "Langfuse keys not configured — tracing disabled. "
            "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable."
        )
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore

        handler = CallbackHandler(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        logger.info("Langfuse tracing enabled → %s", host)
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
