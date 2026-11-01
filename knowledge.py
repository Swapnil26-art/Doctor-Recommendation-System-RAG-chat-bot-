"""PDF -> text chunks. Pure Python, no ML dependencies.

Chunking is per page so every chunk can carry a page number, which the UI uses
to cite sources ("Medical_book.pdf, page 412").
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

_SENTENCE_END = re.compile(r"[.!?]\s")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int
    index: int

    @property
    def record_id(self) -> str:
        """Stable ID, so re-ingesting overwrites instead of duplicating."""
        safe_source = re.sub(r"[^A-Za-z0-9._-]+", "-", self.source)
        return f"{safe_source}:p{self.page}:c{self.index}"

    def as_record(self) -> dict:
        return {
            "_id": self.record_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk": self.index,
        }


def normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    normalized = normalize(text)
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))

    chunks: list[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Look for a sentence break in the second half of the window.
            window_start = start + chunk_size // 2
            matches = list(_SENTENCE_END.finditer(normalized, window_start, end))
            if matches:
                end = matches[-1].end()

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def iter_pdf_chunks(
    pdf_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[Chunk]:
    """Yield chunks from every page of a PDF."""
    from pypdf import PdfReader  # imported lazily: only ingestion needs it

    path = Path(pdf_path)
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page should not fail ingestion
            continue
        for index, piece in enumerate(chunk_text(page_text, chunk_size, overlap)):
            yield Chunk(text=piece, source=path.name, page=page_number, index=index)
