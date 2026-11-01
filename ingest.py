"""Ingest PDFs from ``data/`` into the Pinecone knowledge base.

The index is an *integrated* index (``llama-text-embed-v2``): we send text and
Pinecone computes the embeddings server-side, so no local model is required.

Usage:
    python ingest.py --dry-run   # count chunks, touch nothing
    python ingest.py             # upsert into PINECONE_NAMESPACE
    python ingest.py --reset     # delete the namespace first, then upsert
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import get_settings
from knowledge import iter_pdf_chunks

DATA_PATH = Path("data")

# Pinecone's records API accepts at most 96 records per request.
BATCH_SIZE = 96
MAX_ATTEMPTS = 4

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("ingest")


def load_chunks(data_path: Path, chunk_size: int, overlap: int):
    chunks = []
    pdfs = sorted(data_path.glob("*.pdf"))
    if not pdfs:
        return chunks
    for pdf in pdfs:
        logger.info("Reading %s ...", pdf.name)
        chunks.extend(iter_pdf_chunks(pdf, chunk_size, overlap))
    return chunks


def upsert_in_batches(index, namespace: str, records: list[dict]) -> int:
    total = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                index.upsert_records(namespace=namespace, records=batch)
                break
            except Exception as exc:  # noqa: BLE001 - SDK raises several types
                if attempt == MAX_ATTEMPTS:
                    raise
                wait = 2**attempt
                logger.warning(
                    "Batch %d failed (%s); retrying in %ss", start // BATCH_SIZE + 1, exc, wait
                )
                time.sleep(wait)
        total += len(batch)
        logger.info("Upserted %d/%d chunks", total, len(records))
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone.")
    parser.add_argument("--dry-run", action="store_true", help="only count chunks")
    parser.add_argument(
        "--reset", action="store_true", help="delete the namespace before upserting"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if "PINECONE_API_KEY" in settings.missing_keys:
        logger.error("PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    if not DATA_PATH.is_dir():
        logger.error("Data folder not found: %s", DATA_PATH.resolve())
        return 1

    chunks = load_chunks(DATA_PATH, settings.chunk_size, settings.chunk_overlap)
    logger.info(
        "Prepared %d chunks (chunk_size=%d, overlap=%d)",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if not chunks:
        logger.error("No text found in the PDFs. Are they scanned images?")
        return 1
    if args.dry_run:
        logger.info("Dry run: nothing sent to Pinecone.")
        return 0

    from pinecone import Pinecone

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index)
    namespace = settings.pinecone_namespace

    if args.reset:
        logger.warning("Deleting all records in namespace '%s' ...", namespace)
        index.delete(delete_all=True, namespace=namespace)

    logger.info("Upserting into index '%s' namespace '%s' ...", settings.pinecone_index, namespace)
    try:
        total = upsert_in_batches(index, namespace, [chunk.as_record() for chunk in chunks])
    except Exception:
        logger.exception("Ingestion failed")
        return 1

    logger.info(
        "Done. %d chunks in '%s/%s'. Give Pinecone a few seconds to index them.",
        total,
        settings.pinecone_index,
        namespace,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
