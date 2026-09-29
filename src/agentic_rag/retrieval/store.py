"""Qdrant vector store with hybrid dense/sparse search and tenant scoping."""

import uuid
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

if TYPE_CHECKING:
    from agentic_rag.ingest.models import Chunk
    from agentic_rag.retrieval.embedder import EmbeddingResult

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Fixed namespace so a chunk id always maps to the same point id, across
# processes and across runs.
POINT_NAMESPACE = uuid.UUID("6f1b6b4e-6a1e-4a5a-9f3f-2a1d7c2b4e10")


def point_id(chunk_id: str) -> str:
    """Return a stable, globally unique point id for ``chunk_id``.

    Qdrant point ids must be unique across the whole collection, not within
    one upsert call. Using the enumerate position instead means a second
    upsert starts again at zero and silently overwrites the first call's
    points -- no error, no warning, just missing data that only shows up in
    an audit. Deriving the id from the chunk id also makes re-ingestion
    idempotent: an unchanged chunk overwrites itself rather than duplicating.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


@dataclass(frozen=True)
class SearchHit:
    """A single retrieval result."""

    chunk_id: str
    score: float
    doc_id: str = ""
    relative_id: str = ""
    doc_title: str = ""
    heading_path: tuple[str, ...] = ()
    content: str = ""
    language: str = "en"
    section: str = ""
    source_path: str = ""

    @property
    def citation(self) -> str:
        """Return a human-readable citation for this passage."""
        parts = [self.doc_title, *self.heading_path]
        return " > ".join(p for p in parts if p)


@dataclass
class VectorStore:
    """Qdrant client wrapper for hybrid retrieval."""

    settings: Settings = field(default_factory=get_settings)

    @cached_property
    def client(self) -> Any:
        from qdrant_client import QdrantClient

        config = self.settings.vector_store
        return QdrantClient(
            host=config.host,
            port=config.http_port,
            grpc_port=config.grpc_port,
            prefer_grpc=config.prefer_grpc,
            timeout=config.timeout_seconds,
        )

    @property
    def collection(self) -> str:
        return self.settings.vector_store.collection_name

    def recreate_collection(self) -> None:
        """Drop and recreate the collection with dense and sparse vectors."""
        from qdrant_client import models

        config = self.settings.vector_store
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=config.dense_vector_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        for name in ("language", "section", "tenant", "doc_id", "relative_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        logger.info("collection_created", collection=self.collection)

    def upsert(
        self,
        chunks: "list[Chunk]",
        embeddings: "EmbeddingResult",
        tenant: str = "public",
    ) -> None:
        """Insert chunks with their dense and sparse vectors."""
        from qdrant_client import models

        if len(chunks) != len(embeddings.dense):
            message = f"chunk/embedding length mismatch: {len(chunks)} vs {len(embeddings.dense)}"
            raise ValueError(message)

        batch_size = self.settings.vector_store.upsert_batch_size
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            points = [
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR: embeddings.dense[position],
                        SPARSE_VECTOR: models.SparseVector(
                            indices=list(embeddings.sparse[position].keys()),
                            values=list(embeddings.sparse[position].values()),
                        ),
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "relative_id": chunk.relative_id,
                        "doc_title": chunk.doc_title,
                        "heading_path": chunk.heading_path,
                        "content": chunk.content,
                        "language": chunk.language,
                        "section": (chunk.section_path[0] if chunk.section_path else "other"),
                        "source_path": chunk.source_path,
                        "tenant": tenant,
                    },
                )
                for position, chunk in enumerate(chunks[start:end], start=start)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

        logger.info("upsert_completed", count=len(chunks), tenant=tenant)

    def _filter(
        self,
        tenant: str | None,
        language: str | None,
        sections: tuple[str, ...] = (),
    ) -> Any:
        from qdrant_client import models

        conditions: list[Any] = []
        if tenant is not None:
            conditions.append(
                models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant))
            )
        if language is not None:
            conditions.append(
                models.FieldCondition(key="language", match=models.MatchValue(value=language))
            )
        if sections:
            conditions.append(
                models.FieldCondition(key="section", match=models.MatchAny(any=list(sections)))
            )
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def to_hit(point: Any) -> SearchHit:
        """Convert a Qdrant scored point into a SearchHit."""
        payload = point.payload or {}
        return SearchHit(
            chunk_id=str(payload.get("chunk_id", "")),
            score=float(point.score),
            doc_id=str(payload.get("doc_id", "")),
            relative_id=str(payload.get("relative_id", "")),
            doc_title=str(payload.get("doc_title", "")),
            heading_path=tuple(payload.get("heading_path", ())),
            content=str(payload.get("content", "")),
            language=str(payload.get("language", "")),
            section=str(payload.get("section", "")),
            source_path=str(payload.get("source_path", "")),
        )

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        tenant: str | None = "public",
        language: str | None = None,
        sections: tuple[str, ...] = (),
    ) -> list[SearchHit]:
        """Dense vector search."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR,
            limit=limit,
            query_filter=self._filter(tenant, language, sections),
            with_payload=True,
        )
        return [self.to_hit(p) for p in results.points]

    def search_sparse(
        self,
        weights: dict[int, float],
        limit: int,
        tenant: str | None = "public",
        language: str | None = None,
        sections: tuple[str, ...] = (),
    ) -> list[SearchHit]:
        """Sparse lexical search using learned BGE-M3 term weights."""
        from qdrant_client import models

        if not weights:
            return []
        results = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=list(weights.keys()), values=list(weights.values())),
            using=SPARSE_VECTOR,
            limit=limit,
            query_filter=self._filter(tenant, language, sections),
            with_payload=True,
        )
        return [self.to_hit(p) for p in results.points]

    def count(self) -> int:
        """Return the number of points in the collection."""
        return int(self.client.count(self.collection, exact=True).count)
