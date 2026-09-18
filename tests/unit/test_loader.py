"""Tests for the document loader."""

from pathlib import Path

import pytest

from agentic_rag.ingest.loader import load_documents


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    _write(
        root / "concepts" / "pods.md",
        "---\ntitle: Pods\nweight: 10\n---\n\n" + "Pod content. " * 30,
    )
    _write(
        root / "concepts" / "_index.md",
        "---\ntitle: Concepts\n---\n\n" + "Section landing. " * 30,
    )
    _write(root / "tasks" / "stub.md", "---\ntitle: Stub\n---\n\nToo short.")
    _write(
        root / "includes" / "snippet.md",
        "---\ntitle: Snippet\n---\n\n" + "Reusable snippet. " * 30,
    )
    _write(root / "tasks" / "no-title.md", "Body without frontmatter. " * 30)
    return root


def test_loads_eligible_documents_only(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200)
    doc_ids = {doc.doc_id for doc in documents}
    assert "concepts/pods" in doc_ids
    assert "concepts/_index" not in doc_ids
    assert "includes/snippet" not in doc_ids
    assert "tasks/stub" not in doc_ids


def test_extracts_title_from_frontmatter(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200)
    pods = next(doc for doc in documents if doc.doc_id == "concepts/pods")
    assert pods.title == "Pods"
    assert pods.frontmatter["title"] == "Pods"


def test_falls_back_to_filename_for_title(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200)
    doc = next(d for d in documents if d.doc_id == "tasks/no-title")
    assert doc.title == "No Title"


def test_section_path_reflects_directory_hierarchy(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200)
    pods = next(doc for doc in documents if doc.doc_id == "concepts/pods")
    assert pods.section_path == ["concepts"]


def test_ordering_is_deterministic(docs_root: Path) -> None:
    first = [doc.doc_id for doc in load_documents(docs_root, min_chars=200)]
    second = [doc.doc_id for doc in load_documents(docs_root, min_chars=200)]
    assert first == second


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nonexistent", min_chars=200)
