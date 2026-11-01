"""Tests for PDF chunking (pure Python, no PDF parsing on disk)."""

from __future__ import annotations

import pypdf
import pytest

from knowledge import Chunk, chunk_text, iter_pdf_chunks, normalize


def test_normalize_collapses_whitespace():
    assert normalize("  a\n\n b\t c  ") == "a b c"


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Asthma is a chronic disease.", chunk_size=200) == [
        "Asthma is a chronic disease."
    ]


def test_long_text_is_split_within_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 5
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_text(text, chunk_size=100, overlap=30)

    assert len(chunks) > 3
    first_tail = chunks[0][-20:]
    assert first_tail in chunks[1]


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)


def test_oversized_overlap_is_clamped():
    text = "x" * 500
    chunks = chunk_text(text, chunk_size=100, overlap=500)
    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_record_id_is_stable_and_sanitized():
    chunk = Chunk(text="t", source="Medical book (v2).pdf", page=3, index=7)

    assert chunk.record_id == "Medical-book-v2-.pdf:p3:c7"
    assert chunk.record_id == Chunk("t", "Medical book (v2).pdf", 3, 7).record_id


def test_as_record_contains_text_and_metadata():
    record = Chunk("some text", "book.pdf", 5, 2).as_record()

    assert record == {
        "_id": "book.pdf:p5:c2",
        "text": "some text",
        "source": "book.pdf",
        "page": 5,
        "chunk": 2,
    }


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, path):
        self.pages = [FakePage("First page text."), FakePage(""), FakePage("Third page text.")]


def test_iter_pdf_chunks_adds_page_numbers(monkeypatch):
    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    chunks = list(iter_pdf_chunks("data/Medical_book.pdf", chunk_size=100, overlap=10))

    assert [chunk.page for chunk in chunks] == [1, 3]
    assert all(chunk.source == "Medical_book.pdf" for chunk in chunks)
    assert [chunk.index for chunk in chunks] == [0, 0]


def test_iter_pdf_chunks_survives_unreadable_page(monkeypatch):
    class BrokenPage(FakePage):
        def extract_text(self):
            raise ValueError("corrupt stream")

    class Reader:
        def __init__(self, path):
            self.pages = [BrokenPage("x"), FakePage("Good page.")]

    monkeypatch.setattr(pypdf, "PdfReader", Reader)

    chunks = list(iter_pdf_chunks("book.pdf", chunk_size=100, overlap=10))

    assert len(chunks) == 1
    assert chunks[0].page == 2
