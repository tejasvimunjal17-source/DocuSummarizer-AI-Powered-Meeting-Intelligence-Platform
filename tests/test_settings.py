"""Unit tests for environment-based configuration (no network used)."""
import pytest

from config.settings import ConfigurationError, load_settings, mask_secret


def env(**kw):
    return {k: v for k, v in kw.items()}


def test_valid_gemini_settings():
    s = load_settings(env=env(LLM_PROVIDER="gemini", GEMINI_API_KEY="abcdefgh12345"))
    assert s.provider == "gemini"
    assert s.model  # default model applied


def test_default_provider_is_gemini():
    s = load_settings(env=env(GEMINI_API_KEY="abcdefgh12345"))
    assert s.provider == "gemini"


def test_missing_key_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env=env(LLM_PROVIDER="gemini"))


def test_placeholder_key_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env=env(GEMINI_API_KEY="your_api_key_here"))


def test_unsupported_provider_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env=env(LLM_PROVIDER="foo", GEMINI_API_KEY="abcdefgh12345"))


def test_invalid_temperature_raises():
    with pytest.raises(ConfigurationError):
        load_settings(env=env(GEMINI_API_KEY="abcdefgh12345", LLM_TEMPERATURE="hot"))
    with pytest.raises(ConfigurationError):
        load_settings(env=env(GEMINI_API_KEY="abcdefgh12345", LLM_TEMPERATURE="5"))


def test_api_key_not_exposed_in_repr_or_str():
    key = "SECRETKEY1234567890"
    s = load_settings(env=env(GEMINI_API_KEY=key))
    assert key not in repr(s)
    assert key not in str(s)


def test_mask_secret():
    assert mask_secret(None) == "<not set>"
    assert mask_secret("abc") == "****"
    assert "SECRETKEY" not in mask_secret("SECRETKEY1234567890")
