"""
Vector store facade — vendor-agnostic interface backed by Qdrant.

Why a facade
────────────
The previous code leaked ChromaDB's API directly into ingestion and
retrieval: ``vs.get(where=...)``, ``vs.delete(ids=...)``, ``vs.get(
include=["documents","metadatas"])``. Every call site was coupled to the
vendor. Swapping engines became a multi-file rewrite.

This module exposes a narrow, intent-named interface — ``add_documents``,
``similarity_search``, ``find_by_metadata``, ``delete_by_metadata``,
``all_documents``, ``count`` — that any vector backend can implement.
The default implementation is Qdrant; replacing it with pgvector,
Weaviate, or Milvus is a single-file change.

Why Qdrant (replacing ChromaDB)
───────────────────────────────
For an on-prem / private-network deployment, the choice of vector
database is a procurement decision, not a Python decision. The previous
ChromaDB choice does not survive that review:

  • No authentication primitive (server runs open by default).
  • SQLite-backed with weak concurrent-write guarantees.
  • No payload indexing — per-tenant or per-ACL filters become scans.
  • No snapshot / point-in-time backup primitive.
  • No replication.

Qdrant answers all five. It ships as a single Docker image, supports
API-key auth, has first-class payload indexing for fast metadata
filters, supports snapshots and replication, and exposes both an
HTTP/gRPC server and an in-process embedded mode for development. The
embedded mode is what makes it a viable drop-in for the prototype:
operators can develop against ``QDRANT_PATH=./qdrant_data`` and switch
to ``QDRANT_URL=http://qdrant:6333`` for production with no application
code change.

Deployment modes
────────────────
  Embedded (default):  QDRANT_URL unset, uses QDRANT_PATH on disk.
                       Suitable for single-worker dev / single-tenant
                       small deployments.
  Server   (recommended): QDRANT_URL=http://qdrant:6333 (docker-compose).
                       Required for multi-worker, multi-replica setups
                       and any deployment that wants auth, snapshots,
                       or HA.

Backward-compatibility note
───────────────────────────
``CHROMA_PATH`` from the old config is no longer read by this module.
The audit log written by ingestion/pipeline.py still uses that path for
continuity, but the vector data has moved. The migration is one-shot:
on first ingest after upgrade, Qdrant builds its own collection. The
old ``chroma/`` directory can be deleted once verified.
"""

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from core.config import get_settings
from core.exceptions import VectorStoreError
from services.embedding import get_embeddings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant-backed implementation of the vector store facade.

    Vector size is detected once on first connect by embedding a probe
    string. This avoids hardcoding model dimensions and means swapping
    the embedding provider (or its model) automatically rebuilds the
    collection with the right shape on next startup.

    All payloads carry the chunk text under the ``page_content`` key plus
    whatever metadata the caller attached. Filters use Qdrant's
    ``payload`` indexing — for production deployments you'd add explicit
    payload indexes on ``source_file``, ``doc_hash``, and any other
    frequently-filtered key. We do not create indexes here; the operator
    sizes them based on real cardinality.
    """

    def __init__(self) -> None:
        # Lazy imports keep tests that don't touch the vector store fast.
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        settings = get_settings()
        self._collection = settings.collection_name
        self._embeddings = get_embeddings()

        if settings.qdrant_url:
            logger.info("Connecting to Qdrant server", extra={"url": settings.qdrant_url})
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=10.0,
            )
        else:
            path = Path(settings.qdrant_path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Opening Qdrant embedded store", extra={"path": str(path)})
            self._client = QdrantClient(path=str(path))

        # Probe-and-create on first use. We embed a short string to learn
        # the model's output dimensionality without caching the result —
        # the lru_cache on get_embeddings already memoises the client.
        if not self._client.collection_exists(self._collection):
            probe = self._embeddings.embed_query("__dimension_probe__")
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=len(probe), distance=Distance.COSINE),
            )
            # Payload index on tenant_id makes the per-tenant filter a true
            # index lookup rather than a full scan. Without this, a 100k-
            # document corpus shared across 10 tenants would scan 100k
            # points to return one tenant's slice — the whole multi-tenant
            # story falls over operationally.
            self._ensure_payload_index("tenant_id")
            self._ensure_payload_index("source_file")
            self._ensure_payload_index("doc_hash")
            logger.info(
                "Created Qdrant collection",
                extra={"name": self._collection, "dim": len(probe)},
            )

    def _ensure_payload_index(self, field: str) -> None:
        """Create a payload index for fast metadata filtering.

        Idempotent: Qdrant raises if the index already exists; we swallow
        that specific case so re-creation across restarts is safe.
        """
        from qdrant_client.models import PayloadSchemaType
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            # qdrant-client raises a generic ValueError on "already exists"
            # in embedded mode; we log and continue rather than abort.
            logger.debug("Payload index create skipped", extra={"field": field, "err": str(exc)})

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _point_to_doc(point) -> Document:
        """Translate a Qdrant point/record into a LangChain Document.

        ``page_content`` is pulled out of the payload; everything else
        round-trips as metadata. ``score`` is attached when present
        (Qdrant returns it on similarity search but not on scroll).
        """
        payload = dict(point.payload or {})
        text = payload.pop("page_content", "")
        score = getattr(point, "score", None)
        if score is not None:
            payload["vector_score"] = float(score)
        return Document(page_content=text, metadata=payload)

    def _filter_from_dict(self, where: dict):
        """Translate a plain ``{key: value}`` dict into a Qdrant Filter.

        All conditions are AND-ed (``must=[...]``). For more complex
        predicates (OR / NOT) callers should drop down to the raw
        Qdrant client; we keep the facade narrow on purpose.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        return Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in where.items()
        ])

    # ── Public interface ─────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> List[str]:
        """Embed and upsert a batch of documents. Returns assigned IDs."""
        if not docs:
            return []
        from qdrant_client.models import PointStruct

        try:
            texts = [d.page_content for d in docs]
            vectors = self._embeddings.embed_documents(texts)
            ids = [str(uuid.uuid4()) for _ in docs]
            points = [
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload={"page_content": texts[i], **(docs[i].metadata or {})},
                )
                for i in range(len(docs))
            ]
            self._client.upsert(collection_name=self._collection, points=points)
            return ids
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def similarity_search(
        self,
        query: str,
        k: int,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """Top-k cosine similarity search.

        ``where`` is an optional AND-of-equality filter on payload fields.
        For multi-tenant deployments, callers MUST pass ``{"tenant_id": …}``
        — there is no implicit tenant binding here. The retrieval layer is
        responsible for that policy.
        """
        try:
            vec = self._embeddings.embed_query(query)
            kwargs = {
                "collection_name": self._collection,
                "query": vec,
                "limit": k,
                "with_payload": True,
            }
            if where:
                kwargs["query_filter"] = self._filter_from_dict(where)
            result = self._client.query_points(**kwargs)
            return [self._point_to_doc(p) for p in result.points]
        except Exception as exc:
            raise VectorStoreError(f"Qdrant query failed: {exc}") from exc

    def find_by_metadata(self, where: dict, limit: int = 10_000) -> List[Document]:
        """Return every document whose payload matches all key/value pairs."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._filter_from_dict(where),
                limit=limit,
                with_payload=True,
            )
            return [self._point_to_doc(p) for p in points]
        except Exception as exc:
            raise VectorStoreError(f"Qdrant scroll failed: {exc}") from exc

    def delete_by_metadata(self, where: dict) -> int:
        """Delete every point whose payload matches all key/value pairs.

        Returns the number of points deleted (queried before delete since
        Qdrant's delete response does not include a count).
        """
        from qdrant_client.models import FilterSelector

        try:
            matching = self.find_by_metadata(where)
            if not matching:
                return 0
            self._client.delete(
                collection_name=self._collection,
                points_selector=FilterSelector(filter=self._filter_from_dict(where)),
            )
            return len(matching)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def all_documents(self, batch: int = 1_000) -> List[Document]:
        """Scroll the entire collection. Use sparingly.

        Memory-bounded: yields in batches under the hood, accumulates in
        a list. For corpora past tens of thousands of chunks, prefer
        ``find_by_metadata`` with explicit filters or expose a streaming
        variant. We keep this as a method because BM25 rebuild needs the
        full corpus on every ingest.
        """
        try:
            all_points = []
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=batch,
                    offset=offset,
                    with_payload=True,
                )
                all_points.extend(points)
                if offset is None:
                    break
            return [self._point_to_doc(p) for p in all_points]
        except Exception as exc:
            raise VectorStoreError(f"Qdrant scroll-all failed: {exc}") from exc

    def count(self) -> int:
        try:
            return self._client.count(self._collection, exact=True).count
        except Exception:
            # Fall back to a scroll if count() is unavailable in the
            # embedded build (older qdrant-client revisions).
            return len(self.all_documents())


@lru_cache(maxsize=1)
def get_vectorstore() -> QdrantVectorStore:
    """Return the process-singleton vector store.

    Cached because (a) connection establishment is non-trivial and
    (b) the embedded mode does NOT support multiple concurrent client
    instances on the same path. The cache enforces that invariant.
    """
    return QdrantVectorStore()
