"""Tests for the persisted BM25 index."""

from pathlib import Path
from typing import ClassVar

from agentic_rag.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    build_index,
    stem,
    tokenize,
)


def _doc(chunk_id: str, content: str, **kwargs: object) -> BM25Document:
    defaults = {
        "doc_id": "kubernetes:concepts/pods",
        "relative_id": "concepts/pods",
        "doc_title": "Pods",
        "heading_path": (),
        "language": "en",
        "section": "concepts",
        "tenant": "public",
        "source_path": "concepts/pods.md",
    }
    defaults.update(kwargs)
    return BM25Document(chunk_id=chunk_id, content=content, **defaults)  # type: ignore[arg-type]


def _index(*documents: BM25Document) -> BM25Index:
    index = BM25Index()
    for document in documents:
        index.add(document)
    index.finalise()
    return index


class TestStem:
    def test_inflected_forms_converge_on_one_stem(self) -> None:
        pairs = [
            ("drain", "draining"),
            ("volume", "volumes"),
            ("policy", "policies"),
            ("node", "nodes"),
            ("schedule", "scheduled"),
            ("secret", "secrets"),
            ("pod", "pods"),
            ("service", "services"),
            ("run", "running"),
        ]
        for base, inflected in pairs:
            assert stem(base) == stem(inflected), f"{base} != {inflected}"

    def test_does_not_merge_distinct_terms(self) -> None:
        assert stem("secret") != stem("security")

    def test_leaves_identifiers_untouched(self) -> None:
        assert stem("spec.nodename") == "spec.nodename"
        assert stem("v1.28") == "v1.28"

    def test_leaves_short_tokens_untouched(self) -> None:
        assert stem("tls") == "tls"
        assert stem("api") == "api"

    def test_leaves_cjk_untouched(self) -> None:
        assert stem("排") == "排"

    def test_keeps_genuine_doubled_consonants(self) -> None:
        assert stem("call") == "call"


class TestTokenize:
    def test_lowercases_and_splits_latin(self) -> None:
        assert tokenize("Kubectl DRAIN nodes") == ["kubectl", "drain", "nod"]

    def test_removes_stopwords(self) -> None:
        assert "the" not in tokenize("the node")

    def test_keeps_identifiers_intact(self) -> None:
        assert "spec.nodename" in tokenize("filter on spec.nodeName please")

    def test_splits_chinese_into_characters(self) -> None:
        tokens = tokenize("排空节点")
        assert tokens == ["排", "空", "节", "点"]

    def test_mixed_script_text(self) -> None:
        tokens = tokenize("kubectl 排空 node")
        assert "kubectl" in tokens
        assert "排" in tokens


class TestScoring:
    def test_finds_exact_term(self) -> None:
        index = _index(
            _doc("a", "draining a node evicts its pods safely"),
            _doc("b", "persistent volumes provide durable storage"),
        )
        hits = index.search("drain node", limit=5)
        assert hits[0].chunk_id == "a"

    def test_rare_terms_outrank_common_ones(self) -> None:
        index = _index(
            _doc("common1", "node node node cluster"),
            _doc("common2", "node cluster cluster"),
            _doc("rare", "node cordon uncordon"),
        )
        hits = index.search("cordon", limit=5)
        assert hits[0].chunk_id == "rare"

    def test_unknown_terms_return_nothing(self) -> None:
        index = _index(_doc("a", "draining a node"))
        assert index.search("quantum entanglement", limit=5) == []

    def test_empty_query_returns_nothing(self) -> None:
        index = _index(_doc("a", "draining a node"))
        assert index.search("the and of", limit=5) == []

    def test_empty_index_returns_nothing(self) -> None:
        assert BM25Index().search("anything", limit=5) == []

    def test_title_and_headings_are_searchable(self) -> None:
        index = _index(
            _doc("a", "unrelated body text here", doc_title="Taints and Tolerations"),
            _doc("b", "more unrelated body text"),
        )
        hits = index.search("taints", limit=5)
        assert hits and hits[0].chunk_id == "a"

    def test_results_are_deterministic(self) -> None:
        index = _index(_doc("a", "node drain"), _doc("b", "node drain"))
        first = [h.chunk_id for h in index.search("node drain", limit=5)]
        second = [h.chunk_id for h in index.search("node drain", limit=5)]
        assert first == second

    def test_respects_limit(self) -> None:
        index = _index(*[_doc(str(i), "node drain pods") for i in range(20)])
        assert len(index.search("node", limit=3)) == 3


class TestFiltering:
    def test_tenant_filter_excludes_other_tenants(self) -> None:
        index = _index(
            _doc("public-chunk", "draining a node", tenant="public"),
            _doc("secret-chunk", "draining a node", tenant="restricted"),
        )
        hits = index.search("drain node", limit=10, tenant="public")
        assert [h.chunk_id for h in hits] == ["public-chunk"]

    def test_language_filter(self) -> None:
        index = _index(
            _doc("en", "draining a node", language="en"),
            _doc("zh", "draining a node", language="zh"),
        )
        hits = index.search("drain", limit=10, tenant=None, language="zh")
        assert [h.chunk_id for h in hits] == ["zh"]

    def test_section_filter(self) -> None:
        index = _index(
            _doc("c", "draining a node", section="concepts"),
            _doc("r", "draining a node", section="reference"),
        )
        hits = index.search("drain", limit=10, tenant=None, sections=("concepts",))
        assert [h.chunk_id for h in hits] == ["c"]

    def test_no_tenant_filter_returns_all(self) -> None:
        index = _index(
            _doc("a", "draining a node", tenant="public"),
            _doc("b", "draining a node", tenant="restricted"),
        )
        assert len(index.search("drain", limit=10, tenant=None)) == 2


class TestPersistence:
    def test_roundtrip_preserves_search_results(self, tmp_path: Path) -> None:
        index = _index(
            _doc("a", "draining a node evicts pods"),
            _doc("b", "persistent volume claims bind to volumes"),
        )
        path = tmp_path / "bm25.pkl"
        index.save(path)

        restored = BM25Index.load(path)
        assert restored.size == 2
        assert [h.chunk_id for h in restored.search("drain", limit=5)] == [
            h.chunk_id for h in index.search("drain", limit=5)
        ]

    def test_build_index_from_chunks(self) -> None:
        class FakeChunk:
            chunk_id = "kubernetes:concepts/pods#0"
            doc_id = "kubernetes:concepts/pods"
            relative_id = "concepts/pods"
            doc_title = "Pods"
            heading_path: ClassVar[list[str]] = ["Overview"]
            content = "A pod is the smallest deployable unit."
            language = "en"
            section_path: ClassVar[list[str]] = ["concepts"]
            source_path = "concepts/pods.md"

        index = build_index([FakeChunk()], tenant="public")
        assert index.size == 1
        assert index.search("deployable unit", limit=1)[0].chunk_id == FakeChunk.chunk_id
