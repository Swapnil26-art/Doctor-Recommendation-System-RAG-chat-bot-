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
    assert settings.chunk_size == 1000
    assert settings.places_radius_km == 50
    assert settings.missing_keys == []


def test_missing_keys_lists_names_only():
    settings = Settings(google_api_key="", google_maps_api_key="", pinecone_api_key="")
    assert settings.missing_keys == list(REQUIRED)


def test_partial_configuration_is_detected():
    settings = Settings(google_api_key="a", google_maps_api_key="", pinecone_api_key="c")
    assert settings.missing_keys == ["GOOGLE_MAPS_API_KEY"]


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
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.pinecone_index == "custom-index"
    assert settings.pinecone_namespace == "custom-namespace"
    assert settings.retrieval_top_k == 6
    assert settings.places_radius_km == 10
    assert settings.places_max_results == 3


def test_invalid_integer_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PLACES_RADIUS_KM", "not-a-number")
    get_settings.cache_clear()

    assert get_settings().places_radius_km == 50
