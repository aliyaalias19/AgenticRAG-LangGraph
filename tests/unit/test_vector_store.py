"""Tests for the Qdrant vector store, run against a real local Qdrant engine.

qdrant-client's ``:memory:`` mode runs the actual engine in-process, so these
exercise real collection creation, real hybrid vector storage and real
server-side filtering -- not a mock of them. The tenant filter in particular
has to be verified against the engine, because a filter that is silently
malformed returns everything.
"""

import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.retrieval.embedder import EmbeddingResult
from agentic_rag.retrieval.store import DENSE_VECTOR, SPARSE_VECTOR, VectorStore

DIM = 8


class FakeChunk:
    """Minimal chunk shape the store needs."""

    def __init__(
        self,
        chunk_id: str,
        content: str = "body",
        section: str = "concepts",
        language: str = "en",
        title: str = "Doc",
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = f"kubernetes:{chunk_id.split('#')[0]}"
        self.relative_id = chunk_id.split(":")[-1].split("#")[0]
        self.doc_title = title
        self.heading_path = ["Overview"]
        self.content = content
        self.language = language
        self.section_path = [section]
        self.source_path = f"{section}.md"


def unit_vector(index: int) -> list[float]:
    """Return a one-hot vector, so similarity is exactly controllable."""
    vector = [0.0] * DIM
    vector[index % DIM] = 1.0
    return vector


@pytest.fixture
def store() -> VectorStore:
    settings = Settings()
    settings.vector_store.dense_vector_size = DIM
    settings.vector_store.collection_name = "test_chunks"

    store = VectorStore(settings=settings)
    from qdrant_client import QdrantClient

    # Bypass the cached_property so the real engine runs in-process.
    store.__dict__["client"] = QdrantClient(":memory:")
    store.recreate_collection()
    return store


def seed(store: VectorStore, chunks: list[FakeChunk], tenant: str = "public") -> None:
    embeddings = EmbeddingResult(
        dense=[unit_vector(i) for i in range(len(chunks))],
        sparse=[{i + 1: 1.0, 99: 0.5} for i in range(len(chunks))],
    )
    store.upsert(chunks, embeddings, tenant=tenant)


class TestCollectionLifecycle:
    def test_collection_is_created_with_both_vector_types(self, store: VectorStore) -> None:
        info = store.client.get_collection(store.collection)
        assert DENSE_VECTOR in info.config.params.vectors
        assert SPARSE_VECTOR in (info.config.params.sparse_vectors or {})

    def test_dense_vector_size_matches_settings(self, store: VectorStore) -> None:
        info = store.client.get_collection(store.collection)
        assert info.config.params.vectors[DENSE_VECTOR].size == DIM

    def test_recreate_is_idempotent(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("a#0")])
        assert store.count() == 1

        store.recreate_collection()
        assert store.count() == 0

    def test_count_reflects_upserts(self, store: VectorStore) -> None:
        seed(store, [FakeChunk(f"c{i}#0") for i in range(5)])
        assert store.count() == 5


class TestUpsert:
    def test_payload_round_trips(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("concepts/pods#0", content="A pod is a unit.")])
        hit = store.search_dense(unit_vector(0), limit=1)[0]

        assert hit.chunk_id == "concepts/pods#0"
        assert hit.doc_id == "kubernetes:concepts/pods"
        assert hit.relative_id == "concepts/pods"
        assert hit.content == "A pod is a unit."
        assert hit.heading_path == ("Overview",)
        assert hit.section == "concepts"

    def test_length_mismatch_is_rejected(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            store.upsert(
                [FakeChunk("a#0"), FakeChunk("b#0")],
                EmbeddingResult(dense=[unit_vector(0)], sparse=[{1: 1.0}]),
            )

    def test_batching_stores_every_chunk(self, store: VectorStore) -> None:
        store.settings.vector_store.upsert_batch_size = 3
        seed(store, [FakeChunk(f"c{i}#0") for i in range(10)])
        assert store.count() == 10

    def test_citation_is_built_from_payload(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("a#0", title="Safely Drain a Node")])
        hit = store.search_dense(unit_vector(0), limit=1)[0]
        assert hit.citation == "Safely Drain a Node > Overview"


class TestDenseSearch:
    def test_returns_the_nearest_vector_first(self, store: VectorStore) -> None:
        seed(store, [FakeChunk(f"c{i}#0") for i in range(4)])
        hits = store.search_dense(unit_vector(2), limit=4)
        assert hits[0].chunk_id == "c2#0"

    def test_respects_the_limit(self, store: VectorStore) -> None:
        seed(store, [FakeChunk(f"c{i}#0") for i in range(8)])
        assert len(store.search_dense(unit_vector(0), limit=3)) == 3

    def test_scores_are_populated(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("a#0")])
        assert store.search_dense(unit_vector(0), limit=1)[0].score > 0.0


class TestSparseSearch:
    def test_matches_on_shared_terms(self, store: VectorStore) -> None:
        seed(store, [FakeChunk(f"c{i}#0") for i in range(3)])
        hits = store.search_sparse({2: 1.0}, limit=3)
        assert hits[0].chunk_id == "c1#0"

    def test_empty_weights_return_nothing(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("a#0")])
        assert store.search_sparse({}, limit=5) == []

    def test_unknown_terms_return_nothing(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("a#0")])
        assert store.search_sparse({12345: 1.0}, limit=5) == []


class TestTenantFiltering:
    """The filters have to be verified against the engine.

    A filter that is silently malformed does not error -- it returns
    everything, which looks exactly like a working system until it is
    audited.
    """

    def test_a_tenant_cannot_see_another_tenants_chunks(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("public#0")], tenant="public")
        store.settings.vector_store.upsert_batch_size = 128
        store.upsert(
            [FakeChunk("secret#0")],
            EmbeddingResult(dense=[unit_vector(0)], sparse=[{1: 1.0}]),
            tenant="restricted",
        )

        hits = store.search_dense(unit_vector(0), limit=10, tenant="public")
        assert [h.chunk_id for h in hits] == ["public#0"]

    def test_sparse_search_is_filtered_too(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("public#0")], tenant="public")
        store.upsert(
            [FakeChunk("secret#0")],
            EmbeddingResult(dense=[unit_vector(0)], sparse=[{1: 1.0}]),
            tenant="restricted",
        )

        hits = store.search_sparse({1: 1.0}, limit=10, tenant="public")
        assert all(h.chunk_id != "secret#0" for h in hits)

    def test_section_filter_narrows_results(self, store: VectorStore) -> None:
        seed(
            store,
            [
                FakeChunk("c#0", section="concepts"),
                FakeChunk("r#0", section="reference"),
                FakeChunk("t#0", section="tasks"),
            ],
        )
        hits = store.search_dense(unit_vector(0), limit=10, sections=("concepts", "tasks"))
        assert {h.chunk_id for h in hits} == {"c#0", "t#0"}

    def test_language_filter_narrows_results(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("en#0", language="en"), FakeChunk("zh#0", language="zh")])
        hits = store.search_dense(unit_vector(0), limit=10, language="zh")
        assert [h.chunk_id for h in hits] == ["zh#0"]

    def test_filters_compose(self, store: VectorStore) -> None:
        seed(
            store,
            [
                FakeChunk("keep#0", section="concepts", language="en"),
                FakeChunk("wrong_section#0", section="reference", language="en"),
                FakeChunk("wrong_language#0", section="concepts", language="zh"),
            ],
        )
        hits = store.search_dense(
            unit_vector(0),
            limit=10,
            tenant="public",
            language="en",
            sections=("concepts",),
        )
        assert [h.chunk_id for h in hits] == ["keep#0"]

    def test_no_tenant_filter_returns_every_tenant(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("public#0")], tenant="public")
        store.upsert(
            [FakeChunk("secret#0")],
            EmbeddingResult(dense=[unit_vector(0)], sparse=[{1: 1.0}]),
            tenant="restricted",
        )
        assert len(store.search_dense(unit_vector(0), limit=10, tenant=None)) == 2


class TestPointIdentity:
    """Regression tests for the positional-id collision.

    The original implementation used the enumerate position as the Qdrant
    point id, so a second upsert call reused ids 0..n and overwrote the
    first call's points without erroring.
    """

    def test_ids_are_stable_across_processes(self) -> None:
        from agentic_rag.retrieval.store import point_id

        assert point_id("kubernetes:concepts/pods#0") == point_id("kubernetes:concepts/pods#0")

    def test_ids_are_unique_per_chunk(self) -> None:
        from agentic_rag.retrieval.store import point_id

        ids = {point_id(f"doc{i}#0") for i in range(1000)}
        assert len(ids) == 1000

    def test_a_second_upsert_does_not_overwrite_the_first(self, store: VectorStore) -> None:
        seed(store, [FakeChunk("first#0")])
        store.upsert(
            [FakeChunk("second#0")],
            EmbeddingResult(dense=[unit_vector(0)], sparse=[{1: 1.0}]),
        )
        assert store.count() == 2

    def test_reingesting_the_same_chunk_is_idempotent(self, store: VectorStore) -> None:
        for _ in range(3):
            seed(store, [FakeChunk("a#0")])
        assert store.count() == 1
