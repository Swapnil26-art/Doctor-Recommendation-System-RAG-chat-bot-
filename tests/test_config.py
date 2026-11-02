"""Tests for environment/config handling."""

from __future__ import annotations

import pytest

import config
from config import Settings, get_settings, require_settings

REQUIRED = ("GOOGLE_API_KEY", "GOOGLE_MAPS_API_KEY", "PINECONE_API_KEY")


def test_defaults_are_applied():
    settings = get_settings()
    assert settings.pinecone_index == "medicalbot"
    assert settings.pinecone_namespace == "knowledge-base"
    assert settings.retrieval_top_k == 4
    assert settings.min_relevance_score == 0.30
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_fallback_models == ("gemini-3.6-flash", "gemini-3.5-flash")
    assert settings.chunk_size == 1000
    assert settings.places_radius_km == 50
    assert settings.missing_keys == []


def test_missing_keys_lists_names_only():
    settings = Settings(google_api_key="", google_maps_api_key="", pinecone_api_key="")
    assert settings.missing_keys == list(REQUIRED)


def test_partial_configuration_is_detected():
    settings = Settings(google_api_key="a", google_maps_api_key="", pinecone_api_key="c")
    assert settings.missing_keys == ["GOOGLE_MAPS_API_KEY"]


def test_placeholder_values_are_not_considered_configured():
    # Exactly what ships in .env.example: these must not make /healthz lie.
    settings = Settings(
        google_api_key="YOUR_GEMINI_API_KEY",
        google_maps_api_key="Your_API_Key",
        pinecone_api_key="YOUR_PINECONE_API_KEY",
    )
    assert settings.missing_keys == list(REQUIRED)


@pytest.mark.parametrize(
    "value",
    ["google-key-123", "pinecone-key-123", "test-google-key"],
)
def test_real_looking_keys_are_considered_configured(value):
    assert config.is_placeholder(value) is False


def test_components_only_require_their_own_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_PLACES_API_KEY")
    get_settings.cache_clear()

    # Chat still works with no Maps key ...
    require_settings("GOOGLE_API_KEY", "PINECONE_API_KEY")

    # ... but doctor search must refuse to run.
    with pytest.raises(RuntimeError):
        require_settings("GOOGLE_MAPS_API_KEY")


def test_require_settings_raises_without_keys(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    with pytest.raises(RuntimeError) as excinfo:
        require_settings()

    message = str(excinfo.value)
    for name in REQUIRED:
        assert name in message
    # Values must never appear in the error.
    assert "test-" not in message


def test_dotenv_overrides_stale_system_variables(monkeypatch):
    calls = {}
    monkeypatch.setattr(config, "load_dotenv", lambda **kwargs: calls.update(kwargs))
    get_settings.cache_clear()

    get_settings()

    assert calls == {"override": True}


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX", "custom-index")
    monkeypatch.setenv("PINECONE_NAMESPACE", "custom-namespace")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
    monkeypatch.setenv("PLACES_RADIUS_KM", "10")
    monkeypatch.setenv("PLACES_MAX_RESULTS", "3")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fallback-a, fallback-b ,")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.pinecone_index == "custom-index"
    assert settings.pinecone_namespace == "custom-namespace"
    assert settings.retrieval_top_k == 6
    assert settings.places_radius_km == 10
    assert settings.places_max_results == 3
    assert settings.llm_model == "custom-model"
    assert settings.llm_fallback_models == ("fallback-a", "fallback-b")


def test_empty_fallback_list_is_allowed(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    get_settings.cache_clear()

    assert get_settings().llm_fallback_models == ()


def test_invalid_integer_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PLACES_RADIUS_KM", "not-a-number")
    get_settings.cache_clear()

    assert get_settings().places_radius_km == 50
