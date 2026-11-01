"""Tests for the RAG wrapper.

The real chain needs torch + Pinecone + Gemini, so these tests exercise the
wrapper contract and, importantly, assert that importing it stays cheap.
"""

from __future__ import annotations

import sys

import rag


class FakeChain:
    def __init__(self, payload):
        self.payload = payload
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return self.payload


def test_importing_rag_does_not_pull_in_torch():
    assert "torch" not in sys.modules


def test_answer_question_strips_whitespace(monkeypatch):
    monkeypatch.setattr(rag, "get_qa_chain", lambda: FakeChain({"result": "  hello  "}))
    assert rag.answer_question("question") == "hello"


def test_empty_answer_has_fallback(monkeypatch):
    monkeypatch.setattr(rag, "get_qa_chain", lambda: FakeChain({"result": "   "}))
    assert "could not find" in rag.answer_question("question").lower()


def test_missing_result_key_is_handled(monkeypatch):
    monkeypatch.setattr(rag, "get_qa_chain", lambda: FakeChain({}))
    assert "could not find" in rag.answer_question("question").lower()


def test_chain_is_built_only_once(monkeypatch):
    built = []

    def fake_build():
        built.append(True)
        return FakeChain({"result": "ok"})

    monkeypatch.setattr(rag, "_build_qa_chain", fake_build)
    monkeypatch.setattr(rag, "_qa_chain", None)

    rag.get_qa_chain()
    rag.get_qa_chain()

    assert built == [True]


def test_question_reaches_the_chain(monkeypatch):
    chain = FakeChain({"result": "answer"})
    monkeypatch.setattr(rag, "get_qa_chain", lambda: chain)

    rag.answer_question("What is asthma?")

    assert chain.questions == ["What is asthma?"]


def test_prompt_allows_refusal():
    assert "do not have" in rag.PROMPT_TEMPLATE.lower()
    assert "{context}" in rag.PROMPT_TEMPLATE
    assert "{question}" in rag.PROMPT_TEMPLATE
