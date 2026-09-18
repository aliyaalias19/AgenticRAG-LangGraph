"""Tests for corpus document and manifest models."""

import pytest

from agentic_rag.ingest.models import CorpusManifest, Document


def _make_doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        title=doc_id,
        content=content,
        content_hash=Document.compute_hash(content),
        char_count=len(content),
    )


def test_content_hash_is_deterministic() -> None:
    assert Document.compute_hash("hello") == Document.compute_hash("hello")


def test_content_hash_differs_for_different_content() -> None:
    assert Document.compute_hash("hello") != Document.compute_hash("world")


def test_corpus_hash_is_order_independent() -> None:
    a = _make_doc("a", "alpha")
    b = _make_doc("b", "beta")
    assert CorpusManifest.compute_corpus_hash([a, b]) == (
        CorpusManifest.compute_corpus_hash([b, a])
    )


def test_corpus_hash_changes_when_content_changes() -> None:
    original = [_make_doc("a", "alpha")]
    modified = [_make_doc("a", "alpha modified")]
    assert CorpusManifest.compute_corpus_hash(original) != (
        CorpusManifest.compute_corpus_hash(modified)
    )


def test_char_count_rejects_negative() -> None:
    with pytest.raises(ValueError):
        Document(
            doc_id="a",
            source_path="a.md",
            title="a",
            content="",
            content_hash="x",
            char_count=-1,
        )
