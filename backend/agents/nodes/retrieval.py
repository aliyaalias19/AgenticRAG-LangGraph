"""
Retrieval Node — runs hybrid BM25 + vector search + RRF + reranking.

This node is called on every retrieval attempt (initial and after rewrites).
It delegates to HybridRetriever and attaches retrieval_metadata to state
so the API layer can expose per-stage diagnostic counts to the UI.

Token cost: 0 (no LLM calls — all local computation).
"""

import logging

from langchain_core.documents import Document

from agents.state import RAGState
from core.config import get_settings
from core.exceptions import RetrievalError
from retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)
settings = get_settings()

# Module-level singleton — built once, reused across requests
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(settings)
    return _retriever


def retrieve_node(state: RAGState) -> dict:
    """Execute hybrid retrieval for the current (possibly rewritten) query."""
    query = state.get("rewritten_query") or state["query"]
    steps = state.get("steps_taken", 0)
    tenant_id = state.get("tenant_id") or settings.default_tenant

    try:
        chunks, metadata = _get_retriever().retrieve(query, tenant_id=tenant_id)
    except RetrievalError as exc:
        logger.error("Retrieval failed: %s", exc)
        # Return empty results — the grade node will score 0 → rewrite or answer
        return {
            "retrieved_chunks": [],
            "retrieval_metadata": {"error": str(exc)},
            "steps_taken": steps + 1,
            "reasoning_trace": [f"[Retrieve] FAILED: {exc}"],
        }

    logger.info(
        "Retrieve: %d chunks for %r (dense=%d bm25=%d reranked=%s)",
        len(chunks), query[:60],
        metadata.get("vector_results_count", 0),
        metadata.get("bm25_results_count", 0),
        metadata.get("reranking_applied", False),
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_metadata": metadata,
        "steps_taken": steps + 1,
        "reasoning_trace": [
            f"[Retrieve] {len(chunks)} chunks "
            f"(dense={metadata.get('vector_results_count',0)} "
            f"bm25={metadata.get('bm25_results_count',0)} "
            f"reranked={metadata.get('reranking_applied',False)}) "
            f"· {metadata.get('retrieval_latency_ms',0):.0f}ms"
        ],
    }
