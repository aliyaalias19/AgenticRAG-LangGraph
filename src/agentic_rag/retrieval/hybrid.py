"""Hybrid retrieval: dense + learned-sparse + BM25, fused by RRF and reranked."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_rag.retrieval.store import SearchHit
from agentic_rag.security.tenancy import TenantContext, anonymous_context

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """One retrieval configuration, for controlled comparison."""

    name: str
    use_dense: bool = True
    use_sparse: bool = True
    use_bm25: bool = False
    use_reranker: bool = True
    dense_candidates: int = 50
    sparse_candidates: int = 50
    bm25_candidates: int = 50
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    bm25_weight: float = 1.0
    top_k: int = 10

    @property
    def signals(self) -> tuple[str, ...]:
        """Return the names of the enabled retrieval signals."""
        enabled = []
        if self.use_dense:
            enabled.append("dense")
        if self.use_sparse:
            enabled.append("sparse")
        if self.use_bm25:
            enabled.append("bm25")
        return tuple(enabled)


class EmbedderProtocol(Protocol):
    """Minimal interface the retriever needs from an embedder."""

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]: ...


class StoreProtocol(Protocol):
    """Minimal interface the retriever needs from a vector store."""

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        tenant: str | None = ...,
        language: str | None = ...,
        sections: tuple[str, ...] = ...,
    ) -> list[SearchHit]: ...

    def search_sparse(
        self,
        weights: dict[int, float],
        limit: int,
        tenant: str | None = ...,
        language: str | None = ...,
        sections: tuple[str, ...] = ...,
    ) -> list[SearchHit]: ...


class LexicalProtocol(Protocol):
    """Minimal interface the retriever needs from a BM25 index."""

    def search(
        self,
        query: str,
        limit: int,
        tenant: str | None = ...,
        language: str | None = ...,
        sections: tuple[str, ...] = ...,
    ) -> list[SearchHit]: ...


class RerankProtocol(Protocol):
    """Minimal interface the retriever needs from a reranker."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


DEFAULT_CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("dense_only", use_sparse=False, use_reranker=False),
    RetrievalConfig(
        "bm25_only", use_dense=False, use_sparse=False, use_bm25=True, use_reranker=False
    ),
    RetrievalConfig("sparse_only", use_dense=False, use_reranker=False),
    RetrievalConfig("hybrid_rrf", use_bm25=True, use_reranker=False),
    RetrievalConfig("hybrid_rrf_rerank", use_bm25=True, use_reranker=True),
    RetrievalConfig(
        "hybrid_rrf_rerank_deep",
        use_bm25=True,
        use_reranker=True,
        dense_candidates=100,
        sparse_candidates=100,
        bm25_candidates=100,
    ),
)


@dataclass
class HybridRetriever:
    """Runs a retrieval configuration end to end under a tenant scope."""

    embedder: EmbedderProtocol
    store: StoreProtocol
    reranker: RerankProtocol
    lexical: LexicalProtocol | None = None
    settings: Settings = field(default_factory=get_settings)

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
        context: TenantContext | None = None,
        language: str | None = None,
    ) -> list[SearchHit]:
        """Retrieve chunks for ``query`` under the given configuration and scope.

        The tenant filter is applied inside each backend query, not by
        post-filtering results. Post-filtering would leak information through
        result counts and would silently shrink the candidate pool.
        """
        context = context or anonymous_context(self.settings)
        sections = context.allowed_sections

        rankings: list[list[SearchHit]] = []
        weights: list[float] = []

        if config.use_dense or config.use_sparse:
            dense_vector, sparse_vector = self.embedder.encode_query(query)

            if config.use_dense:
                rankings.append(
                    self.store.search_dense(
                        dense_vector,
                        config.dense_candidates,
                        context.tenant_id,
                        language,
                        sections,
                    )
                )
                weights.append(config.dense_weight)

            if config.use_sparse:
                rankings.append(
                    self.store.search_sparse(
                        sparse_vector,
                        config.sparse_candidates,
                        context.tenant_id,
                        language,
                        sections,
                    )
                )
                weights.append(config.sparse_weight)

        if config.use_bm25 and self.lexical is not None:
            rankings.append(
                self.lexical.search(
                    query,
                    config.bm25_candidates,
                    context.tenant_id,
                    language,
                    sections,
                )
            )
            weights.append(config.bm25_weight)

        if not rankings:
            return []

        fused = (
            rankings[0]
            if len(rankings) == 1
            else reciprocal_rank_fusion(rankings, k=config.rrf_k, weights=weights)
        )

        if config.use_reranker:
            pool = fused[: self.settings.reranker.candidate_pool]
            results = self.reranker.rerank(query, pool, config.top_k)
        else:
            results = fused[: config.top_k]

        logger.info(
            "retrieval_completed",
            config=config.name,
            signals=",".join(config.signals),
            candidates=len(fused),
            returned=len(results),
            **context.as_log_fields(),
        )
        return results


def build_retriever(settings: Settings | None = None) -> HybridRetriever:
    """Construct a retriever with the production components wired in."""
    from agentic_rag.retrieval.bm25 import INDEX_FILENAME, BM25Index
    from agentic_rag.retrieval.embedder import Embedder
    from agentic_rag.retrieval.reranker import CrossEncoderReranker
    from agentic_rag.retrieval.store import VectorStore

    settings = settings or get_settings()
    index_path = Path(settings.paths.processed_dir) / INDEX_FILENAME
    lexical = BM25Index.load(index_path) if index_path.is_file() else None

    return HybridRetriever(
        embedder=Embedder(settings=settings),
        store=VectorStore(settings=settings),
        reranker=CrossEncoderReranker(settings=settings),
        lexical=lexical,
        settings=settings,
    )
