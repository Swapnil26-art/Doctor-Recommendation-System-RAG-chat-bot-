"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from config import get_settings


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch):
    """Give every test a complete (but fake) set of API keys."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-maps-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
