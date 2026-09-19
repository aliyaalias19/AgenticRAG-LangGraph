"""Tests for stratified document sampling."""

from agentic_rag.eval.sampling import allocate_targets, sample_documents
from agentic_rag.ingest.models import Document

WEIGHTS = {
    "concepts": 0.30,
    "tasks": 0.30,
    "reference": 0.25,
    "tutorials": 0.10,
    "setup": 0.05,
}


def _make_doc(doc_id: str, section: str, chars: int = 5000) -> Document:
    content = "x" * chars
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        title=doc_id,
        content=content,
        content_hash=Document.compute_hash(content),
        char_count=chars,
        section_path=[section],
        language="en",
    )


class TestAllocateTargets:
    def test_allocations_sum_to_total(self) -> None:
        assert sum(allocate_targets(150, WEIGHTS).values()) == 150

    def test_small_totals_still_sum_exactly(self) -> None:
        assert sum(allocate_targets(10, WEIGHTS).values()) == 10

    def test_respects_relative_weights(self) -> None:
        targets = allocate_targets(100, WEIGHTS)
        assert targets["concepts"] == 30
        assert targets["setup"] == 5

    def test_is_deterministic(self) -> None:
        assert allocate_targets(37, WEIGHTS) == allocate_targets(37, WEIGHTS)


class TestSampleDocuments:
    def _corpus(self) -> list[Document]:
        return [_make_doc(f"{section}/doc{i}", section) for section in WEIGHTS for i in range(50)]

    def test_returns_requested_total(self) -> None:
        sampled = sample_documents(
            self._corpus(),
            total=50,
            section_weights=WEIGHTS,
            min_chars=1000,
            max_chars=12000,
            seed=42,
        )
        assert len(sampled) == 50

    def test_is_reproducible_with_the_same_seed(self) -> None:
        kwargs = {
            "total": 20,
            "section_weights": WEIGHTS,
            "min_chars": 1000,
            "max_chars": 12000,
        }
        first = sample_documents(self._corpus(), seed=42, **kwargs)  # type: ignore[arg-type]
        second = sample_documents(self._corpus(), seed=42, **kwargs)  # type: ignore[arg-type]
        assert [d.doc_id for d in first] == [d.doc_id for d in second]

    def test_different_seeds_differ(self) -> None:
        kwargs = {
            "total": 20,
            "section_weights": WEIGHTS,
            "min_chars": 1000,
            "max_chars": 12000,
        }
        first = sample_documents(self._corpus(), seed=1, **kwargs)  # type: ignore[arg-type]
        second = sample_documents(self._corpus(), seed=2, **kwargs)  # type: ignore[arg-type]
        assert [d.doc_id for d in first] != [d.doc_id for d in second]

    def test_excludes_documents_outside_size_bounds(self) -> None:
        corpus = [
            _make_doc("concepts/tiny", "concepts", chars=100),
            _make_doc("concepts/huge", "concepts", chars=50000),
            _make_doc("concepts/good", "concepts", chars=5000),
        ]
        sampled = sample_documents(
            corpus,
            total=3,
            section_weights={"concepts": 1.0},
            min_chars=1000,
            max_chars=12000,
            seed=42,
        )
        assert [d.doc_id for d in sampled] == ["concepts/good"]

    def test_filters_by_language(self) -> None:
        english = _make_doc("concepts/en", "concepts")
        chinese = _make_doc("concepts/zh", "concepts")
        chinese.language = "zh"

        sampled = sample_documents(
            [english, chinese],
            total=2,
            section_weights={"concepts": 1.0},
            min_chars=1000,
            max_chars=12000,
            seed=42,
            language="en",
        )
        assert [d.doc_id for d in sampled] == ["concepts/en"]
