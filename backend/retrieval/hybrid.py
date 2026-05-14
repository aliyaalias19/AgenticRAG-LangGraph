"""
Hybrid retrieval: BM25 + dense vector search + RRF fusion + cross-encoder reranking.

Architecture decision: why hybrid?
─────────────────────────────────
  Sparse (BM25):  strong on exact keywords, product codes, named entities.
  Dense (vector): strong on semantic paraphrase, abbreviations, synonyms.
  RRF fusion:     combines rank positions from both without requiring
                  normalised scores — robust to score distribution differences.
  Reranking:      cross-encoder's joint (query, doc) scoring further improves
                  precision without re-embedding everything.

The four-stage pipeline (BM25 → vector → RRF → rerank) is the current
industry best practice for enterprise RAG (see: Cohere, Weaviate, LlamaIndex
hybrid search papers).

Tradeoff vs pure vector:
  +Higher recall on keyword-heavy queries
  +Better precision after reranking
  -Slightly higher latency (BM25 index lookup + reranker inference)
  For the use case (enterprise policy docs), this tradeoff clearly favours hybrid.
"""

import logging
import time
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from core.config import Settings, get_settings
from infrastructure.circuit_breaker import CircuitBreaker
from infrastructure.prometheus import RETRIEVAL_LATENCY
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import get_vectorstore

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score.

    k=60 is the standard default from the original RRF paper (Cormack 2009).
    Higher k smooths rank differences; lower k amplifies them.
    """
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """Stateful retriever that orchestrates the full retrieval pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._bm25 = bm25 or BM25Retriever(self._settings)
        self._reranker = reranker or CrossEncoderReranker(
            model_id=self._settings.reranker_model
        )
        self._vs_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        self._bm25.load_from_disk()

    def rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current vector store contents.

        Called after ingestion so the sparse index stays in sync with
        dense storage without a full restart.
        """
        docs = get_vectorstore().all_documents()
        if docs:
            self._bm25.build([d.page_content for d in docs], [d.metadata for d in docs])

    def retrieve(
        self,
        query: str,
        tenant_id: str | None = None,
    ) -> Tuple[List[Document], dict]:
        """Run the full hybrid retrieval pipeline.

        ``tenant_id`` partitions every retriever call. Both the dense
        vector search (server-side filter) and the BM25 search
        (client-side over-retrieve + filter) honour it. Passing None
        falls back to the configured default tenant — never silently
        cross-tenant.

        Returns (chunks, metadata_dict) where metadata captures per-stage
        counts and latency for observability.
        """
        t0 = time.time()
        settings = self._settings
        tenant = tenant_id or settings.default_tenant
        where = {"tenant_id": tenant}

        # ── Stage 1: Dense vector search ──────────────────────────────────────
        try:
            vs = self._vs_breaker.call(get_vectorstore)
            dense = vs.similarity_search(query, settings.vector_top_k, where=where)
        except Exception as exc:
            from core.exceptions import RetrievalError
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        # ── Stage 2: Sparse BM25 search ───────────────────────────────────────
        sparse = self._bm25.retrieve(query, top_k=settings.bm25_top_k, where=where)

        # ── Stage 3: RRF fusion ───────────────────────────────────────────────
        scores: Dict[str, Tuple[float, Document]] = {}
        for rank, doc in enumerate(dense):
            key = doc.page_content[:100]
            prev_score = scores.get(key, (0.0, doc))[0]
            doc.metadata["vector_rank"] = rank
            scores[key] = (prev_score + _rrf_score(rank), doc)

        for rank, doc in enumerate(sparse):
            key = doc.page_content[:100]
            prev_score = scores.get(key, (0.0, doc))[0]
            doc.metadata["bm25_rank"] = rank
            scores[key] = (prev_score + _rrf_score(rank), doc)

        fused = [
            doc for _, doc in sorted(
                scores.values(), key=lambda x: x[0], reverse=True
            )
        ][: settings.final_top_k]

        # Attach RRF score to metadata for observability
        for rrf_rank, doc in enumerate(fused):
            doc.metadata["rrf_score"] = _rrf_score(rrf_rank)

        # ── Stage 4: Cross-encoder reranking ──────────────────────────────────
        reranking_applied = settings.reranking_enabled and len(fused) > 1
        if reranking_applied:
            fused = self._reranker.rerank(query, fused)

        latency_ms = (time.time() - t0) * 1000
        RETRIEVAL_LATENCY.labels(tenant=tenant).observe(latency_ms / 1000.0)

        metadata = {
            "query_used": query,
            "tenant_id": tenant,
            "vector_results_count": len(dense),
            "bm25_results_count": len(sparse),
            "fused_results_count": len(fused),
            "reranking_applied": reranking_applied,
            "retrieval_latency_ms": round(latency_ms, 1),
        }
        logger.debug(
            "Hybrid retrieval: %d dense + %d sparse → %d fused (%.0fms)",
            len(dense), len(sparse), len(fused), latency_ms,
        )
        return fused, metadata
