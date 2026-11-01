"""Tests for the ingestion batching/retry logic (no Pinecone calls)."""

from __future__ import annotations

import pytest

import ingest
from config import get_settings


class FakeIndex:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.batches: list[tuple[str, list[dict]]] = []

    def upsert_records(self, namespace, records):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("429 too many requests")
        self.batches.append((namespace, records))


def records(count: int) -> list[dict]:
    return [{"_id": f"id-{i}", "text": "x"} for i in range(count)]


def test_batches_respect_the_96_record_limit():
    index = FakeIndex()

    total = ingest.upsert_in_batches(index, "kb", records(200))

    assert total == 200
    assert [len(batch) for _namespace, batch in index.batches] == [96, 96, 8]
    assert all(namespace == "kb" for namespace, _batch in index.batches)


def test_empty_input_makes_no_requests():
    index = FakeIndex()

    assert ingest.upsert_in_batches(index, "kb", []) == 0
    assert index.batches == []


def test_retries_a_throttled_batch(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)
    index = FakeIndex(failures=2)

    total = ingest.upsert_in_batches(index, "kb", records(3))

    assert total == 3
    assert len(index.batches) == 1


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)
    index = FakeIndex(failures=999)

    with pytest.raises(RuntimeError):
        ingest.upsert_in_batches(index, "kb", records(1))


def test_dry_run_does_not_touch_pinecone(monkeypatch, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(ingest, "DATA_PATH", tmp_path)
    monkeypatch.setattr(
        ingest,
        "load_chunks",
        lambda path, size, overlap: [type("C", (), {"as_record": lambda self: {}})()],
    )

    def explode(*args, **kwargs):
        raise AssertionError("Pinecone must not be used during a dry run")

    import pinecone

    monkeypatch.setattr(pinecone, "Pinecone", explode)

    assert ingest.main(["--dry-run"]) == 0


def test_dry_run_needs_no_credentials(monkeypatch, tmp_path):
    (tmp_path / "book.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(ingest, "DATA_PATH", tmp_path)
    monkeypatch.setattr(
        ingest,
        "load_chunks",
        lambda path, size, overlap: [type("C", (), {"as_record": lambda self: {}})()],
    )
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    get_settings.cache_clear()

    assert ingest.main(["--dry-run"]) == 0


def test_real_run_requires_a_pinecone_key(monkeypatch, tmp_path):
    (tmp_path / "book.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(ingest, "DATA_PATH", tmp_path)
    monkeypatch.setattr(ingest, "load_chunks", lambda path, size, overlap: ["chunk"])
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    get_settings.cache_clear()

    assert ingest.main([]) == 1
