"""
Centralized application settings.

All environment variables are read exactly once here.
Every other module imports from this file — no raw os.getenv() elsewhere.

Usage
-----
from agent.config import get_settings

settings = get_settings()
model_name = settings.llm_model
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Resolve .env relative to the project root (two levels up from this file)
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Pydantic-settings handles:
    - Type coercion (str → float, etc.)
    - .env file loading
    - Validation with clear error messages on startup
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown env vars — don't crash on unrelated vars
    )

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    llm_provider: str = "openai"
    """'openai' for OpenAI hosted API; 'openai_compatible' for self-hosted."""

    # OpenAI
    openai_api_key: str = ""
    """Required when llm_provider='openai'."""

    # OpenAI-compatible (Ollama, vLLM, LM Studio, etc.)
    llm_base_url: str = ""
    """Required when llm_provider='openai_compatible'. E.g. http://localhost:11434/v1"""

    llm_api_key: str = "ollama"
    """API key for self-hosted provider (Ollama ignores this; vLLM may require it)."""

    # Shared LLM settings
    llm_model: str = "gpt-4o-2024-08-06"
    """Model identifier. Defaults to GPT-4o; override for self-hosted (e.g. llama3.1:70b)."""

    llm_temperature: float = 0.0
    """Inference temperature. 0 = fully deterministic (recommended for credit decisions)."""

    llm_max_tokens: int = 8192
    """
    Token limit for the LLM response.
    Maps to max_tokens (gpt-4o, gpt-4, self-hosted) or
    max_completion_tokens (o1/o3/o4, gpt-5) — selected automatically in build_llm().
    For gpt-5 / o-series, build_llm floors this at 8192 so reasoning tokens
    do not exhaust the budget before a visible JSON verdict is produced.
    """

    # ------------------------------------------------------------------
    # Agent behaviour
    # ------------------------------------------------------------------
    agent_timeout_seconds: int = 60
    """Wall-clock timeout for a single agent assessment run."""

    agent_max_retries: int = 3
    """Number of tenacity retry attempts on transient LLM errors."""

    # ------------------------------------------------------------------
    # Langfuse observability
    # ------------------------------------------------------------------
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_path: str = "./Companies.parquet"
    """Path to Companies.parquet. Resolved relative to CWD at runtime."""

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalise_provider(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError(f"llm_temperature must be between 0 and 2, got {v}")
        return v

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "Settings":
        """Fail at startup if the chosen provider is missing required credentials."""
        if self.llm_provider == "openai" and not self.openai_api_key.strip():
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.\n"
                "Get your key from https://platform.openai.com/api-keys"
            )
        if self.llm_provider == "openai_compatible" and not self.llm_base_url.strip():
            raise ValueError(
                "LLM_PROVIDER=openai_compatible requires LLM_BASE_URL to be set.\n"
                "Example: LLM_BASE_URL=http://localhost:11434/v1"
            )
        if self.llm_provider not in {"openai", "openai_compatible"}:
            raise ValueError(
                f"LLM_PROVIDER='{self.llm_provider}' is not valid.\n"
                "Valid values: 'openai', 'openai_compatible'"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_secret_key.strip() and self.langfuse_public_key.strip())

    @property
    def is_openai(self) -> bool:
        return self.llm_provider == "openai"

    @property
    def is_openai_compatible(self) -> bool:
        return self.llm_provider == "openai_compatible"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached with lru_cache — loaded once at first call, reused everywhere.
    In tests, call get_settings.cache_clear() before patching env vars.
    """
    settings = Settings()
    logger.debug(
        "Settings loaded: provider=%s model=%s data_path=%s langfuse=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.data_path,
        settings.langfuse_enabled,
    )
    return settings
