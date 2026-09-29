"""Cross-encoder reranking with BGE-reranker-v2-m3."""

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Protocol

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger
from agentic_rag.retrieval.store import SearchHit

logger = get_logger(__name__)


class RerankerProtocol(Protocol):
    """Minimal interface a reranker must satisfy."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranker.

    A bi-encoder embeds query and passage independently, so it can never model
    their interaction; it only ever compares two fixed points in space. A
    cross-encoder scores the pair jointly through full attention, which is much
    more accurate and much more expensive -- hence reranking a shortlist rather
    than the whole corpus.
    """

    settings: Settings = field(default_factory=get_settings)

    @cached_property
    def _model(self) -> Any:
        from FlagEmbedding import FlagReranker

        config = self.settings.reranker
        logger.info("reranker_loading", model=config.model_name)
        model = FlagReranker(
            config.model_name,
            use_fp16=config.use_fp16,
            device=config.device,
            cache_dir=str(self.settings.embedding.cache_dir),
        )
        logger.info("reranker_loaded", model=config.model_name)
        return model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Rescore hits against the query and return the best ``top_k``."""
        if not hits:
            return []

        pairs = [[query, hit.content] for hit in hits]
        raw = self._model.compute_score(
            pairs, batch_size=self.settings.reranker.batch_size, normalize=True
        )
        scores = raw if isinstance(raw, list) else [raw]

        rescored = [
            replace(hit, score=float(score)) for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: (-h.score, h.chunk_id))
        return rescored[:top_k]


@dataclass
class NullReranker:
    """Pass-through reranker used when reranking is disabled."""

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Return the first ``top_k`` hits unchanged."""
        del query
        return hits[:top_k]
