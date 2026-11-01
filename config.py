"""Application settings loaded from the environment (``.env`` supported).

Keeping configuration in one place means the rest of the code never reads
``os.environ`` directly and never has to know about API-key names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

DEFAULT_INDEX_NAME = "medicalbot"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_PLACES_RADIUS_KM = 50
DEFAULT_PLACES_MAX_RESULTS = 5


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    google_api_key: str
    google_maps_api_key: str
    pinecone_api_key: str
    pinecone_index: str = DEFAULT_INDEX_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    places_radius_km: int = DEFAULT_PLACES_RADIUS_KM
    places_max_results: int = DEFAULT_PLACES_MAX_RESULTS
    log_level: str = "INFO"

    @property
    def missing_keys(self) -> list[str]:
        """Names (never values) of required keys that are not configured."""
        required = {
            "GOOGLE_API_KEY": self.google_api_key,
            "GOOGLE_MAPS_API_KEY": self.google_maps_api_key,
            "PINECONE_API_KEY": self.pinecone_api_key,
        }
        return [name for name, value in required.items() if not value]


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings, loading ``.env`` on first use."""
    load_dotenv()
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index=os.getenv("PINECONE_INDEX", DEFAULT_INDEX_NAME),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        places_radius_km=_int_from_env("PLACES_RADIUS_KM", DEFAULT_PLACES_RADIUS_KM),
        places_max_results=_int_from_env("PLACES_MAX_RESULTS", DEFAULT_PLACES_MAX_RESULTS),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def require_settings() -> Settings:
    """Return settings, or raise if a required API key is missing.

    Called lazily (at request time) rather than at import time so the web app
    can still boot and serve ``/healthz`` while it is being misconfigured.
    """
    settings = get_settings()
    if settings.missing_keys:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(settings.missing_keys)
        )
    return settings
