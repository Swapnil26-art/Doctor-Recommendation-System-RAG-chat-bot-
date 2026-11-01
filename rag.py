"""Retrieval-Augmented Generation pipeline.

Heavy dependencies (torch, transformers, Pinecone client, Gemini) are imported
lazily so that importing the web app stays fast and unit tests do not need the
full ~2 GB ML stack installed.
"""

from __future__ import annotations

import logging
import threading

from config import require_settings

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are MedBot, a careful medical information assistant.
Use the following pieces of context from a medical reference book to answer the
user's question. If the answer is not in the context, say that you do not have
that information in your knowledge base. Do not invent an answer.

Context:
{context}

Question:
{question}

Helpful Answer (concise, factual, and written for a non-specialist):
"""

# How many chunks to retrieve for each question.
RETRIEVER_K = 3

_lock = threading.Lock()
_qa_chain = None


def _build_qa_chain():
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore

    settings = require_settings()

    logger.info("Loading embedding model %s", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    logger.info("Connecting to Pinecone index %s", settings.pinecone_index)
    vectorstore = PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
    )

    logger.info("Initialising LLM %s", settings.llm_model)
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def get_qa_chain():
    """Return the process-wide RAG chain, building it on first use."""
    global _qa_chain
    if _qa_chain is None:
        with _lock:
            if _qa_chain is None:
                _qa_chain = _build_qa_chain()
                logger.info("RAG chain is ready")
    return _qa_chain


def answer_question(question: str) -> str:
    """Answer a medical question from the vector knowledge base."""
    result = get_qa_chain().invoke(question)
    answer = (result.get("result") or "").strip()
    if not answer:
        return "I could not find an answer to that in the medical knowledge base."
    return answer
