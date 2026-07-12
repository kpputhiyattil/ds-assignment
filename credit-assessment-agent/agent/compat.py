"""
LangChain 0.2+ compatibility shim.

LangChain 0.2 removed several module-level attributes that older versions of
langgraph, langfuse, langchain_community, and other ecosystem packages still
reference. This module patches them back onto the langchain namespace before
any of those packages are imported.

Import this module ONCE, as early as possible in the process entry point
(e.g. the top of streamlit_app.py), before any langchain sub-package imports.

References
----------
https://python.langchain.com/docs/versions/migrating_chains/
"""

from __future__ import annotations

import logging
import types

logger = logging.getLogger(__name__)

# Attributes removed in langchain 0.2 that ecosystem packages still access.
# Maps attribute_name → safe default value.
_MISSING_ATTRS: dict[str, object] = {
    "debug": False,
    "verbose": False,
    "llm_cache": None,
    # callback_manager was replaced by langchain_core.callbacks — stub with None
    "callback_manager": None,
}


def apply() -> None:
    """
    Patch all known removed attributes onto the langchain module.

    Safe to call multiple times — subsequent calls are no-ops for already-patched
    attributes. Logs a DEBUG message for each patch applied.
    """
    import langchain  # imported here so this module can be imported pre-langchain

    patched: list[str] = []
    for attr, default in _MISSING_ATTRS.items():
        if not hasattr(langchain, attr):
            setattr(langchain, attr, default)
            patched.append(attr)

    if patched:
        logger.debug(
            "langchain compat shim: patched missing attributes %s onto langchain %s",
            patched,
            getattr(langchain, "__version__", "unknown"),
        )


# Auto-apply on import so callers only need `import agent.compat`
apply()
