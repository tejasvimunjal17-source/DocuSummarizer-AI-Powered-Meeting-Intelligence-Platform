"""Provider-agnostic factory that creates a LangChain chat model.

Only this file knows which provider package to use, so adding a provider later
does not require changing the rest of the application. Provider packages are
imported lazily so a missing optional package gives a clear error message.

NOTE (Day 1): this factory is implemented but has NOT been tested with a real
API call. Prompt chains that use it come on Day 4.
"""
from __future__ import annotations

from typing import Any, Optional

from config.settings import ConfigurationError, Settings, load_settings


def build_llm(settings: Optional[Settings] = None) -> Any:
    """Create a LangChain chat model for the configured provider."""
    settings = settings or load_settings()

    if settings.provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ConfigurationError(
                "Package 'langchain-google-genai' is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=settings.api_key,
            temperature=settings.temperature,
        )

    if settings.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "Package 'langchain-openai' is not installed. "
                "Run: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
        )

    raise ConfigurationError(f"Unsupported provider '{settings.provider}'.")
