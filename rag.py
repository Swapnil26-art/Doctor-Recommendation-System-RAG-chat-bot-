"""Retrieval-Augmented Generation.

Retrieval uses Pinecone's hosted embedding model (the index is an *integrated*
index, so Pinecone embeds the question server-side). Generation uses Gemini.

There is no local model to load: torch, sentence-transformers and LangChain are
no longer needed, which keeps startup fast and the image small.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from config import require_settings

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are MedBot, a careful medical information assistant.
Answer the question using ONLY the context below, which comes from a medical
reference book. If the context does not contain the answer, say that you do not
have that information in your knowledge base. Never invent facts, dosages, or
diagnoses. Write for a non-specialist in at most 200 words.

Context:
{context}

Question:
{question}

Answer:"""

# Keeps prompts (and Gemini cost) bounded regardless of what retrieval returns.
MAX_CONTEXT_CHARS = 8000

# Gemini occasionally answers "503 UNAVAILABLE / high demand" or rate-limits us.
# Those are worth retrying; auth and bad-request errors are not.
GENERATION_ATTEMPTS = 2  # per model, before falling back to the next one
RETRY_BASE_DELAY_SECONDS = 1.0
# Without an explicit timeout the HTTP client can wait forever and pin a worker.
GENERATION_TIMEOUT_MS = 20_000
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_STATUS_NAMES = frozenset(
    {"DEADLINE_EXCEEDED", "INTERNAL", "RESOURCE_EXHAUSTED", "UNAVAILABLE"}
)


@dataclass(frozen=True)
class Source:
    """A retrieved passage, used for citations in the UI."""

    source: str
    page: int | None
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


_index = None
_genai_client = None
_lock = threading.Lock()


def reset_clients() -> None:
    """Drop cached clients (used by tests)."""
    global _index, _genai_client
    with _lock:
        _index = None
        _genai_client = None


def _get_index():
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                from pinecone import Pinecone

                settings = require_settings("PINECONE_API_KEY")
                logger.info(
                    "Connecting to Pinecone index %s (namespace %s)",
                    settings.pinecone_index,
                    settings.pinecone_namespace,
                )
                _index = Pinecone(api_key=settings.pinecone_api_key).Index(
                    settings.pinecone_index
                )
    return _index


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        with _lock:
            if _genai_client is None:
                from google import genai

                settings = require_settings("GOOGLE_API_KEY")
                logger.info("Initialising Gemini client for %s", settings.llm_model)
                _genai_client = genai.Client(
                    api_key=settings.google_api_key,
                    http_options={"timeout": GENERATION_TIMEOUT_MS},
                )
    return _genai_client


def retrieve(question: str, top_k: int | None = None) -> list[tuple[str, Source]]:
    """Return ``(text, Source)`` pairs, most similar first."""
    settings = require_settings("PINECONE_API_KEY")
    index = _get_index()

    response = index.search(
        namespace=settings.pinecone_namespace,
        query={
            "inputs": {"text": question},
            "top_k": top_k or settings.retrieval_top_k,
        },
        fields=["text", "source", "page"],
    )
    payload = response.to_dict() if hasattr(response, "to_dict") else response
    hits = (payload.get("result") or {}).get("hits") or []
    logger.info("Retrieved %d chunks for question", len(hits))

    results: list[tuple[str, Source]] = []
    for hit in hits:
        fields = hit.get("fields") or {}
        text = str(fields.get("text") or "").strip()
        if not text:
            continue
        score = float(hit.get("_score") or 0.0)
        if score < settings.min_relevance_score:
            continue
        page = fields.get("page")
        results.append(
            (
                text,
                Source(
                    source=str(fields.get("source") or "knowledge base"),
                    page=int(page) if isinstance(page, (int, float)) else None,
                    score=round(score, 4),
                ),
            )
        )
    return results


def build_prompt(question: str, chunks: list[tuple[str, Source]]) -> str:
    context_parts: list[str] = []
    used = 0
    for text, _source in chunks:
        if used + len(text) > MAX_CONTEXT_CHARS:
            text = text[: max(0, MAX_CONTEXT_CHARS - used)]
        if not text:
            break
        context_parts.append(text)
        used += len(text)
    return PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(context_parts), question=question)


def _is_transient(exc: BaseException) -> bool:
    """True for errors that a retry might cure (throttling, outage).

    ``google.genai`` errors expose an integer HTTP ``code`` and a string
    ``status``; other SDKs only expose one of the two.
    """
    try:
        if int(getattr(exc, "code", None)) in TRANSIENT_HTTP_CODES:
            return True
    except (TypeError, ValueError):
        pass
    status = getattr(exc, "status", None)
    return isinstance(status, str) and status.upper() in TRANSIENT_STATUS_NAMES


def _is_timeout(exc: BaseException) -> bool:
    """True when the model timed out, so retrying it just wastes the budget."""
    try:
        if int(getattr(exc, "code", None)) == 504:
            return True
    except (TypeError, ValueError):
        pass
    return str(getattr(exc, "status", "")).upper() == "DEADLINE_EXCEEDED"


def _model_plan(settings) -> list[tuple[str, int]]:
    """Primary model gets a retry; fallbacks get one shot each."""
    plan = [(settings.llm_model, GENERATION_ATTEMPTS)]
    plan.extend((model, 1) for model in settings.llm_fallback_models)
    return plan


def generate(prompt: str) -> str:
    settings = require_settings("GOOGLE_API_KEY")
    client = _get_genai_client()

    last_error: Exception | None = None
    for model, attempts in _model_plan(settings):
        for attempt in range(1, attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt
                )
                if model != settings.llm_model:
                    logger.info("Answered with fallback model %s", model)
                return (getattr(response, "text", "") or "").strip()
            except Exception as exc:
                last_error = exc
                if not _is_transient(exc):
                    raise
                if attempt < attempts and not _is_timeout(exc):
                    delay = RETRY_BASE_DELAY_SECONDS * attempt
                    logger.warning(
                        "%s failed (%s); retrying in %.1fs (attempt %d/%d)",
                        model,
                        exc,
                        delay,
                        attempt,
                        attempts,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "%s unavailable (%s); trying the next model", model, exc
                    )
                    break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini is unavailable")  # pragma: no cover


def _dedupe_sources(chunks: list[tuple[str, Source]]) -> list[Source]:
    best: dict[tuple[str, int | None], Source] = {}
    for _text, source in chunks:
        key = (source.source, source.page)
        if key not in best or source.score > best[key].score:
            best[key] = source
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def answer_question(question: str) -> Answer:
    """Answer a medical question from the vector knowledge base."""
    chunks = retrieve(question)
    if not chunks:
        return Answer(
            text=(
                "I couldn't find anything about that in my medical knowledge base. "
                "Try rephrasing the question, or ask about a specific condition."
            ),
            sources=[],
        )

    prompt = build_prompt(question, chunks)
    # Defensive: a whitespace-only completion must fall back, not render blank.
    answer = (generate(prompt) or "").strip()
    if not answer:
        answer = "I couldn't generate an answer for that. Please try again."

    return Answer(text=answer, sources=_dedupe_sources(chunks))
