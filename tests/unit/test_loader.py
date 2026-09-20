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
    documents = load_documents(docs_root, min_chars=200, source_name="k8s")
    doc_ids = {doc.doc_id for doc in documents}
    assert "k8s:concepts/pods" in doc_ids
    assert "k8s:concepts/_index" not in doc_ids
    assert "k8s:includes/snippet" not in doc_ids
    assert "k8s:tasks/stub" not in doc_ids


def test_extracts_title_from_frontmatter(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200, source_name="k8s")
    pods = next(doc for doc in documents if doc.doc_id == "k8s:concepts/pods")
    assert pods.title == "Pods"
    assert pods.frontmatter["title"] == "Pods"


def test_falls_back_to_filename_for_title(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200, source_name="k8s")
    doc = next(d for d in documents if d.doc_id == "k8s:tasks/no-title")
    assert doc.title == "No Title"


def test_section_path_reflects_directory_hierarchy(docs_root: Path) -> None:
    documents = load_documents(docs_root, min_chars=200, source_name="k8s")
    pods = next(doc for doc in documents if doc.doc_id == "k8s:concepts/pods")
    assert pods.section_path == ["concepts"]


def test_ordering_is_deterministic(docs_root: Path) -> None:
    first = [doc.doc_id for doc in load_documents(docs_root, min_chars=200)]
    second = [doc.doc_id for doc in load_documents(docs_root, min_chars=200)]
    assert first == second


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nonexistent", min_chars=200)


def test_clean_markdown_strips_shortcodes_and_comments() -> None:
    from agentic_rag.ingest.loader import clean_markdown

    raw = "<!-- overview -->\n\n{{% thirdparty-content %}}\n\nReal content here."
    assert clean_markdown(raw) == "Real content here."


def test_clean_markdown_keeps_text_inside_shortcodes() -> None:
    from agentic_rag.ingest.loader import clean_markdown

    raw = "{{< note >}}\nImportant detail.\n{{< /note >}}"
    assert "Important detail." in clean_markdown(raw)


def test_relative_id_is_shared_across_sources(docs_root: Path) -> None:
    english = load_documents(docs_root, min_chars=200, source_name="k8s")
    chinese = load_documents(docs_root, min_chars=200, source_name="k8s-zh")

    en_pods = next(d for d in english if d.relative_id == "concepts/pods")
    zh_pods = next(d for d in chinese if d.relative_id == "concepts/pods")

    assert en_pods.doc_id != zh_pods.doc_id
    assert en_pods.relative_id == zh_pods.relative_id
