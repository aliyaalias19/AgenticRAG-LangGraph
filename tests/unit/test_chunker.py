"""Tests for heading-aware document chunking."""

from agentic_rag.ingest.chunker import (
    _merge_undersized,
    _split_oversized,
    chunk_document,
    split_into_sections,
)
from agentic_rag.ingest.models import Document

CHUNK_KWARGS = {
    "max_chars": 1200,
    "overlap_chars": 150,
    "min_chars": 100,
    "max_heading_depth": 3,
}


def _make_document(content: str, doc_id: str = "concepts/pods") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        title="Pods",
        content=content,
        content_hash=Document.compute_hash(content),
        char_count=len(content),
        section_path=["concepts"],
    )


class TestSectionSplitting:
    def test_splits_on_headings(self) -> None:
        text = "## First\n\nAlpha content.\n\n## Second\n\nBeta content."
        sections = split_into_sections(text, max_depth=3)

        assert len(sections) == 2
        assert sections[0].heading_path == ["First"]
        assert "Alpha content." in sections[0].content
        assert sections[1].heading_path == ["Second"]

    def test_preserves_heading_hierarchy(self) -> None:
        text = "## Parent\n\nIntro.\n\n### Child\n\nDetail."
        sections = split_into_sections(text, max_depth=3)

        assert sections[0].heading_path == ["Parent"]
        assert sections[1].heading_path == ["Parent", "Child"]

    def test_pops_stack_on_shallower_heading(self) -> None:
        text = "## A\n\nOne.\n\n### A1\n\nTwo.\n\n## B\n\nThree."
        sections = split_into_sections(text, max_depth=3)

        assert sections[-1].heading_path == ["B"]

    def test_ignores_hashes_inside_code_fences(self) -> None:
        text = (
            "## Real Heading\n\n"
            "```bash\n"
            "# This is a shell comment, not a heading\n"
            "kubectl get pods\n"
            "```\n\n"
            "Trailing prose."
        )
        sections = split_into_sections(text, max_depth=3)

        assert len(sections) == 1
        assert sections[0].heading_path == ["Real Heading"]
        assert "shell comment" in sections[0].content

    def test_headings_deeper_than_max_depth_stay_inline(self) -> None:
        text = "## Shown\n\nAlpha.\n\n#### Too Deep\n\nBeta."
        sections = split_into_sections(text, max_depth=3)

        assert len(sections) == 1
        assert "#### Too Deep" in sections[0].content

    def test_content_before_first_heading_is_kept(self) -> None:
        text = "Preamble text.\n\n## Heading\n\nBody."
        sections = split_into_sections(text, max_depth=3)

        assert sections[0].heading_path == []
        assert "Preamble" in sections[0].content

    def test_document_without_headings_yields_one_section(self) -> None:
        sections = split_into_sections("Just prose, no headings.", max_depth=3)

        assert len(sections) == 1
        assert sections[0].heading_path == []

    def test_strips_hugo_anchor_ids_from_headings(self) -> None:
        text = "## Using cgroup v2 {#using-cgroupv2}\n\nBody text here."
        sections = split_into_sections(text, max_depth=3)

        assert sections[0].heading_path == ["Using cgroup v2"]


class TestSizeSplitting:
    def test_short_text_is_not_split(self) -> None:
        assert _split_oversized("short", max_chars=1200, overlap_chars=150) == ["short"]

    def test_long_text_is_split_within_bounds(self) -> None:
        text = "\n\n".join(f"Paragraph number {i}. " * 10 for i in range(40))
        parts = _split_oversized(text, max_chars=1200, overlap_chars=150)

        assert len(parts) > 1
        assert all(len(part) <= 1200 for part in parts)

    def test_single_huge_paragraph_is_hard_split(self) -> None:
        parts = _split_oversized("x" * 5000, max_chars=1200, overlap_chars=150)

        assert len(parts) > 1
        assert all(len(part) <= 1200 for part in parts)

    def test_undersized_parts_are_merged(self) -> None:
        merged = _merge_undersized(["a" * 500, "tiny", "b" * 500], min_chars=100)

        assert len(merged) == 2
        assert "tiny" in merged[0]


class TestChunkDocument:
    def test_chunk_ids_are_sequential_and_stable(self) -> None:
        document = _make_document(
            "## A\n\n" + "Alpha content. " * 12 + "\n\n## B\n\n" + "Beta content. " * 12
        )
        chunks = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]

        assert [c.chunk_id for c in chunks] == ["concepts/pods#0", "concepts/pods#1"]
        assert [c.chunk_index for c in chunks] == [0, 1]

    def test_chunks_inherit_document_metadata(self) -> None:
        document = _make_document("## A\n\n" + "Alpha content here. " * 10)
        chunk = chunk_document(document, **CHUNK_KWARGS)[0]  # type: ignore[arg-type]

        assert chunk.doc_id == "concepts/pods"
        assert chunk.doc_title == "Pods"
        assert chunk.section_path == ["concepts"]

    def test_contextual_text_prefixes_title_and_headings(self) -> None:
        document = _make_document(
            "## Parent\n\n"
            + "Parent intro text. " * 10
            + "\n\n### Child\n\n"
            + "Detail text here. " * 10
        )
        chunks = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]
        child = next(c for c in chunks if c.heading_path == ["Parent", "Child"])

        assert child.contextual_text.startswith("Pods > Parent > Child\n\n")
        assert "Detail text here." in child.contextual_text

    def test_no_chunk_exceeds_max_chars(self) -> None:
        body = "\n\n".join(f"Sentence {i} with filler text. " * 8 for i in range(30))
        document = _make_document(f"## Big Section\n\n{body}")
        chunks = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]

        assert all(c.char_count <= 1200 for c in chunks)

    def test_chunks_are_never_empty(self) -> None:
        document = _make_document("## A\n\n\n\n## B\n\nReal content here.")
        chunks = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]

        assert all(c.content.strip() for c in chunks)

    def test_chunking_is_deterministic(self) -> None:
        document = _make_document("## A\n\nAlpha.\n\n## B\n\nBeta.")
        first = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]
        second = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]

        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.content for c in first] == [c.content for c in second]

    def test_chunks_below_min_chars_are_dropped(self) -> None:
        document = _make_document("## A\n\nx\n\n## B\n\n" + "Real content. " * 20)
        chunks = chunk_document(document, **CHUNK_KWARGS)  # type: ignore[arg-type]

        assert all(c.char_count >= 100 for c in chunks)
