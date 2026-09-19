"""Tests for corpus persistence and manifest construction."""

from pathlib import Path

from agentic_rag.config.settings import Settings
from agentic_rag.ingest.models import Document, SourceRecord
from agentic_rag.ingest.pipeline import (
    build_manifest,
    read_corpus,
    write_corpus,
    write_manifest,
)


def _make_doc(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        title=doc_id,
        content=content,
        content_hash=Document.compute_hash(content),
        char_count=len(content),
        section_path=["concepts"],
    )


def _make_source_record(name: str = "kubernetes") -> SourceRecord:
    return SourceRecord(
        name=name,
        repo_url="https://example.com/repo.git",
        repo_ref="main",
        commit_sha="abc123",
        docs_subpath="content/en/docs",
        language="en",
        document_count=1,
    )


def test_corpus_roundtrip_preserves_documents(tmp_path: Path) -> None:
    documents = [_make_doc("a", "alpha"), _make_doc("b", "beta")]
    path = tmp_path / "corpus.jsonl"

    write_corpus(documents, path)
    restored = read_corpus(path)

    assert [d.doc_id for d in restored] == ["a", "b"]
    assert restored[0].content == "alpha"
    assert restored[0].section_path == ["concepts"]


def test_corpus_file_has_one_line_per_document(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    write_corpus([_make_doc("a", "alpha"), _make_doc("b", "beta")], path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_manifest_totals_match_documents() -> None:
    documents = [_make_doc("a", "alpha"), _make_doc("b", "beta")]
    sources = [_make_source_record()]
    manifest = build_manifest(documents, [], sources, Settings())

    assert manifest.document_count == 2
    assert manifest.total_chars == len("alpha") + len("beta")
    assert manifest.sources[0].commit_sha == "abc123"


def test_manifest_is_written_as_readable_json(tmp_path: Path) -> None:
    manifest = build_manifest([_make_doc("a", "alpha")], [], [_make_source_record()], Settings())
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)
    content = path.read_text(encoding="utf-8")

    assert '"commit_sha": "abc123"' in content
    assert content.endswith("\n")


def test_manifest_records_multiple_sources() -> None:
    sources = [_make_source_record("kubernetes"), _make_source_record("kubernetes-zh")]
    manifest = build_manifest([_make_doc("a", "alpha")], [], sources, Settings())

    assert [s.name for s in manifest.sources] == ["kubernetes", "kubernetes-zh"]
