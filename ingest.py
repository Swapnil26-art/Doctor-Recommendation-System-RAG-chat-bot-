"""One-off ingestion: PDFs in ``data/`` -> chunks -> embeddings -> Pinecone.

Usage:
    python ingest.py

Requires PINECONE_API_KEY and the ML dependencies (requirements-ml.txt).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

DATA_PATH = Path("data")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("ingest")


def load_pdf_files(data_path: Path):
    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def main() -> int:
    settings = get_settings()
    if "PINECONE_API_KEY" in settings.missing_keys:
        logger.error("PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    if not DATA_PATH.is_dir() or not any(DATA_PATH.glob("*.pdf")):
        logger.error("No PDF files found in %s", DATA_PATH.resolve())
        return 1

    logger.info("Loading documents from %s ...", DATA_PATH.resolve())
    documents = load_pdf_files(DATA_PATH)
    logger.info("Loaded %d pages.", len(documents))

    chunks = split_documents(documents)
    logger.info("Split into %d chunks.", len(chunks))
    if not chunks:
        logger.error("The PDFs produced no text chunks. Are they scanned images?")
        return 1

    logger.info("Loading embedding model %s ...", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    logger.info("Upserting into Pinecone index '%s' ...", settings.pinecone_index)
    try:
        PineconeVectorStore.from_documents(
            chunks,
            embeddings,
            index_name=settings.pinecone_index,
        )
    except Exception:
        logger.exception("Ingestion failed")
        logger.error(
            "Check that the index '%s' exists with dimension 384 and metric cosine.",
            settings.pinecone_index,
        )
        return 1

    logger.info("Successfully ingested %d chunks into '%s'.", len(chunks), settings.pinecone_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
