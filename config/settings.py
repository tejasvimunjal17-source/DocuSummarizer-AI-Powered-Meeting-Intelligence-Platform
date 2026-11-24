"""Secure, environment-based configuration for DocuSummarizer.

API keys are read ONLY from environment variables (or a local .env file that
is excluded from Git). They are never hardcoded and never printed in full.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from dotenv import load_dotenv

SUPPORTED_PROVIDERS = ("gemini", "openai")

# Environment variable that holds the API key for each provider.
API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Default models. Override with LLM_MODEL in .env if a model name changes.
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
}

DEFAULT_PROVIDER = "gemini"
DEFAULT_TEMPERATURE = 0.0

# Obvious placeholder values that must be rejected as "not a real key".
_PLACEHOLDER_KEYS = {"", "your_api_key_here", "changeme", "your-api-key"}


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


def mask_secret(secret: Optional[str]) -> str:
    """Return a safe-to-log version of a secret (never the full value)."""
    if not secret:
        return "<not set>"
    if len(secret) <= 8:
        return "****"
    return f"{secret[:2]}****{secret[-2:]}"


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    provider: str
    api_key: str = field(repr=False)  # repr=False keeps the key out of logs
    model: str
    temperature: float = DEFAULT_TEMPERATURE

    def __repr__(self) -> str:
        return (
            f"Settings(provider={self.provider!r}, model={self.model!r}, "
            f"temperature={self.temperature}, api_key={mask_secret(self.api_key)!r})"
        )

    __str__ = __repr__


def load_settings(
    env: Optional[Mapping[str, str]] = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Load and validate settings.

    Args:
        env: Optional mapping to read from instead of os.environ (used by tests).
        load_dotenv_file: If True and env is None, load a local .env file first.

    Raises:
        ConfigurationError: with a helpful message if anything is wrong.
    """
    if env is None:
        if load_dotenv_file:
            load_dotenv()  # does not override variables already set
        env = os.environ

    provider = env.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    key_var = API_KEY_ENV_VARS[provider]
    api_key = env.get(key_var, "").strip()
    if api_key.lower() in _PLACEHOLDER_KEYS:
        raise ConfigurationError(
            f"{key_var} is missing or still a placeholder. Copy .env.example to "
            f".env and put your real {provider} API key there. "
            "Never commit the .env file."
        )

    model = env.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]

    raw_temp = env.get("LLM_TEMPERATURE", "").strip()
    if raw_temp:
        try:
            temperature = float(raw_temp)
        except ValueError as exc:
            raise ConfigurationError(
                f"LLM_TEMPERATURE must be a number, got '{raw_temp}'."
            ) from exc
        if not 0.0 <= temperature <= 2.0:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0.0 and 2.0.")
    else:
        temperature = DEFAULT_TEMPERATURE

    return Settings(
        provider=provider, api_key=api_key, model=model, temperature=temperature
    )
