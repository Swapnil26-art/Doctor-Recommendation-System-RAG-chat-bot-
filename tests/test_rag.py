"""Tests for the RAG wrapper (retrieval shape, prompting, citations).

No network and no model downloads: the Pinecone index and Gemini client are
replaced with fakes.
"""

from __future__ import annotations

import sys

import pytest

import rag
from config import get_settings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class FakeIndex:
    def __init__(self, payload=None):
        self.payload = payload or {"result": {"hits": []}}
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.payload)


def make_chunks():
    return [
        ("Asthma is a chronic disease of the airways.", rag.Source("Medical_book.pdf", 12, 0.91)),
        ("Wheezing and cough are common symptoms.", rag.Source("Medical_book.pdf", 13, 0.84)),
    ]


def test_importing_rag_does_not_pull_in_torch():
    assert "torch" not in sys.modules


def test_retrieve_maps_hits_and_skips_empty_text(monkeypatch):
    index = FakeIndex(
        {
            "result": {
                "hits": [
                    {
                        "_id": "a",
                        "_score": 0.83456,
                        "fields": {
                            "text": "Asthma is a chronic disease.",
                            "source": "Medical_book.pdf",
                            "page": 12,
                        },
                    },
                    {"_id": "b", "_score": 0.2, "fields": {"text": "   "}},
                ]
            }
        }
    )
    monkeypatch.setattr(rag, "_get_index", lambda: index)

    results = rag.retrieve("what is asthma", top_k=2)

    assert len(results) == 1
    text, source = results[0]
    assert text == "Asthma is a chronic disease."
    assert source == rag.Source("Medical_book.pdf", 12, 0.8346)

    call = index.calls[0]
    assert call["namespace"] == get_settings().pinecone_namespace
    assert call["query"] == {"inputs": {"text": "what is asthma"}, "top_k": 2}
    assert call["fields"] == ["text", "source", "page"]


def test_retrieve_drops_chunks_below_the_relevance_floor(monkeypatch):
    index = FakeIndex(
        {
            "result": {
                "hits": [
                    {"_id": "a", "_score": 0.62, "fields": {"text": "relevant"}},
                    {"_id": "b", "_score": 0.29, "fields": {"text": "irrelevant"}},
                ]
            }
        }
    )
    monkeypatch.setattr(rag, "_get_index", lambda: index)

    results = rag.retrieve("question")

    assert [text for text, _source in results] == ["relevant"]


def test_all_weak_matches_mean_no_answer(monkeypatch):
    index = FakeIndex(
        {"result": {"hits": [{"_id": "a", "_score": 0.12, "fields": {"text": "noise"}}]}}
    )
    monkeypatch.setattr(rag, "_get_index", lambda: index)
    monkeypatch.setattr(rag, "generate", lambda prompt: "should not be called")

    answer = rag.answer_question("something not in the book")

    assert "couldn't find" in answer.text.lower()
    assert answer.sources == []


def test_retrieve_uses_configured_top_k(monkeypatch):
    index = FakeIndex()
    monkeypatch.setattr(rag, "_get_index", lambda: index)

    rag.retrieve("question")

    assert index.calls[0]["query"]["top_k"] == get_settings().retrieval_top_k


def test_retrieve_handles_missing_source_metadata(monkeypatch):
    index = FakeIndex(
        {"result": {"hits": [{"_id": "a", "_score": 0.5, "fields": {"text": "Some text"}}]}}
    )
    monkeypatch.setattr(rag, "_get_index", lambda: index)

    _text, source = rag.retrieve("q")[0]

    assert source.source == "knowledge base"
    assert source.page is None


def test_prompt_contains_question_and_context():
    prompt = rag.build_prompt("What is asthma?", make_chunks())

    assert "What is asthma?" in prompt
    assert "Asthma is a chronic disease" in prompt
    assert "ONLY the context" in prompt


def test_prompt_context_is_capped():
    huge = [("x" * 5000, rag.Source("book.pdf", 1, 0.9)) for _ in range(10)]
    prompt = rag.build_prompt("q", huge)

    assert len(prompt) < rag.MAX_CONTEXT_CHARS + 2000


def test_sources_are_deduped_by_page(monkeypatch):
    chunks = [
        ("first half", rag.Source("book.pdf", 10, 0.7)),
        ("second half", rag.Source("book.pdf", 10, 0.9)),
        ("other page", rag.Source("book.pdf", 11, 0.8)),
    ]
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k=None: chunks)
    monkeypatch.setattr(rag, "generate", lambda prompt: "answer")

    sources = rag.answer_question("q").sources

    assert len(sources) == 2
    assert sources[0] == rag.Source("book.pdf", 10, 0.9)
    assert sources[0].score >= sources[1].score


def test_answer_uses_generation(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k=None: make_chunks())
    monkeypatch.setattr(rag, "generate", lambda prompt: "  Asthma affects the airways.  ")

    answer = rag.answer_question("what is asthma")

    assert answer.text == "Asthma affects the airways."
    assert len(answer.sources) == 2


def test_no_context_skips_the_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k=None: [])
    monkeypatch.setattr(rag, "generate", lambda prompt: calls.append(prompt) or "x")

    answer = rag.answer_question("something unknown")

    assert calls == []
    assert "couldn't find" in answer.text.lower()
    assert answer.sources == []


def test_empty_generation_uses_fallback(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k=None: make_chunks())
    monkeypatch.setattr(rag, "generate", lambda prompt: "   ")

    answer = rag.answer_question("q")

    assert "couldn't generate" in answer.text.lower()


def test_retrieve_errors_propagate(monkeypatch):
    def boom(question, top_k=None):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(rag, "retrieve", boom)

    with pytest.raises(RuntimeError):
        rag.answer_question("q")


class FakeGeminiError(Exception):
    """Mimics google.genai.errors.APIError's code/status shape."""

    def __init__(self, code=None, status=None):
        super().__init__(f"{code} {status}")
        self.code = code
        self.status = status


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append({"model": model, "contents": contents})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return type("Response", (), {"text": outcome})()


class FakeGeminiClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def test_generate_retries_transient_failures_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    client = FakeGeminiClient(
        [
            FakeGeminiError(None, "RESOURCE_EXHAUSTED"),  # status-only error
            "  Asthma affects the airways.  ",
        ]
    )
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    assert rag.generate("prompt") == "Asthma affects the airways."
    assert len(client.models.calls) == 2
    assert sleeps == [1.0]


def test_generate_falls_back_when_the_primary_model_is_down(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    settings = get_settings()
    fallback = settings.llm_fallback_models[0]
    client = FakeGeminiClient(
        [
            FakeGeminiError(503, "UNAVAILABLE"),
            FakeGeminiError(504, "DEADLINE_EXCEEDED"),
            "answer from the fallback",
        ]
    )
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    assert rag.generate("prompt") == "answer from the fallback"
    assert [call["model"] for call in client.models.calls] == [
        settings.llm_model,
        settings.llm_model,
        fallback,
    ]
    assert sleeps == [1.0]


def test_generate_skips_the_retry_when_the_primary_times_out(monkeypatch):
    """A 20s timeout must fail over immediately, not burn two timeouts."""
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    settings = get_settings()
    client = FakeGeminiClient(
        [FakeGeminiError(504, "DEADLINE_EXCEEDED"), "answer from the fallback"]
    )
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    assert rag.generate("prompt") == "answer from the fallback"
    assert [call["model"] for call in client.models.calls] == [
        settings.llm_model,
        settings.llm_fallback_models[0],
    ]
    assert sleeps == []


def test_generate_does_not_retry_client_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    client = FakeGeminiClient([FakeGeminiError(400, "INVALID_ARGUMENT")])
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    with pytest.raises(FakeGeminiError):
        rag.generate("prompt")

    assert sleeps == []
    assert len(client.models.calls) == 1


def test_generate_gives_up_after_all_models_fail(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag.time, "sleep", sleeps.append)
    plan = rag._model_plan(get_settings())
    total_calls = sum(attempts for _model, attempts in plan)
    client = FakeGeminiClient([FakeGeminiError(503, "UNAVAILABLE")] * total_calls)
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    with pytest.raises(FakeGeminiError):
        rag.generate("prompt")

    assert len(client.models.calls) == total_calls
    assert sleeps == [1.0]


def test_chat_does_not_require_a_maps_key(monkeypatch):
    """The chatbot must keep working when only doctor search is unconfigured."""
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    get_settings.cache_clear()
    client = FakeGeminiClient(["answer"])
    monkeypatch.setattr(rag, "_get_genai_client", lambda: client)

    assert rag.generate("prompt") == "answer"


def test_genai_client_is_created_with_a_request_timeout(monkeypatch):
    import sys
    import types

    import google

    captured = {}

    class FakeGenai:
        def Client(self, **kwargs):
            captured.update(kwargs)
            return object()

    fake_module = types.ModuleType("google.genai")
    fake_module.Client = FakeGenai().Client
    monkeypatch.setitem(sys.modules, "google.genai", fake_module)
    monkeypatch.setattr(google, "genai", fake_module, raising=False)
    rag.reset_clients()
    try:
        rag._get_genai_client()
    finally:
        rag.reset_clients()

    assert captured["http_options"]["timeout"] == rag.GENERATION_TIMEOUT_MS
