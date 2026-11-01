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
# The knowledge base lives in its own namespace so it can be rebuilt without
# touching anything else in the index.
DEFAULT_NAMESPACE = "knowledge-base"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"
DEFAULT_TOP_K = 4
# Below this cosine score the chunks are noise; better to admit we don't know.
DEFAULT_MIN_SCORE = 0.30
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_PLACES_RADIUS_KM = 50
DEFAULT_PLACES_MAX_RESULTS = 5

# Values shipped in .env.example. Treating them as "configured" makes
# /healthz report ok while every request fails with 401/403.
_PLACEHOLDER_MARKERS = (
    "your_",
    "your-",
    "api_key",
    "api-key",
    "replace-me",
    "changeme",
    "<",
    "xxx",
)


def is_placeholder(value: str) -> bool:
    """True for empty values and for the examples in ``.env.example``."""
    lowered = (value or "").strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    google_api_key: str
    google_maps_api_key: str
    pinecone_api_key: str
    pinecone_index: str = DEFAULT_INDEX_NAME
    pinecone_namespace: str = DEFAULT_NAMESPACE
    llm_model: str = DEFAULT_LLM_MODEL
    retrieval_top_k: int = DEFAULT_TOP_K
    min_relevance_score: float = DEFAULT_MIN_SCORE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
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
        return [name for name, value in required.items() if is_placeholder(value)]


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _float_from_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings, loading ``.env`` on first use.

    ``override=True`` means the project's ``.env`` wins over a stale
    ``PINECONE_API_KEY`` left behind in the OS environment. Containers have no
    ``.env``, so real environment variables still take effect there.
    """
    load_dotenv(override=True)
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index=os.getenv("PINECONE_INDEX", DEFAULT_INDEX_NAME),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        retrieval_top_k=_int_from_env("RETRIEVAL_TOP_K", DEFAULT_TOP_K),
        min_relevance_score=_float_from_env("MIN_RELEVANCE_SCORE", DEFAULT_MIN_SCORE),
        chunk_size=_int_from_env("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_int_from_env("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        places_radius_km=_int_from_env("PLACES_RADIUS_KM", DEFAULT_PLACES_RADIUS_KM),
        places_max_results=_int_from_env("PLACES_MAX_RESULTS", DEFAULT_PLACES_MAX_RESULTS),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def require_settings(*names: str) -> Settings:
    """Return settings, or raise if one of ``names`` is not configured.

    Callers ask only for the keys they actually use: a missing Maps key must
    not break the chatbot, and a missing Gemini key must not break ingestion.
    With no arguments every required key is checked.
    """
    settings = get_settings()
    fields = {
        "GOOGLE_API_KEY": settings.google_api_key,
        "GOOGLE_MAPS_API_KEY": settings.google_maps_api_key,
        "PINECONE_API_KEY": settings.pinecone_api_key,
    }
    missing = [
        name for name in (names or tuple(fields)) if is_placeholder(fields[name])
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return settings
