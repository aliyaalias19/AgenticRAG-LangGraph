"""Heading-aware chunking of markdown documents."""

import re
from dataclasses import dataclass, field

from agentic_rag.ingest.models import Chunk, Document
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CODE_FENCE_PATTERN = re.compile(r"^```", re.MULTILINE)
PARAGRAPH_SEPARATOR = "\n\n"
HEADING_ANCHOR_PATTERN = re.compile(r"\s*\{#[^}]*\}\s*$")


@dataclass
class Section:
    """A contiguous region of a document under a heading hierarchy."""

    heading_path: list[str] = field(default_factory=list)
    content: str = ""


def _fence_line_numbers(text: str) -> set[int]:
    """Return line indices that sit inside fenced code blocks."""
    inside: set[int] = set()
    open_fence = False
    for index, line in enumerate(text.splitlines()):
        if line.lstrip().startswith("```"):
            open_fence = not open_fence
            inside.add(index)
            continue
        if open_fence:
            inside.add(index)
    return inside


def split_into_sections(text: str, max_depth: int) -> list[Section]:
    """Split markdown into sections using headings up to ``max_depth``."""
    lines = text.splitlines()
    fenced = _fence_line_numbers(text)

    sections: list[Section] = []
    current = Section()
    heading_stack: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        match = HEADING_PATTERN.match(line) if index not in fenced else None

        if match is None:
            current.content += line + "\n"
            continue

        level = len(match.group(1))
        title = HEADING_ANCHOR_PATTERN.sub("", match.group(2)).strip()

        if level > max_depth:
            current.content += line + "\n"
            continue

        if current.content.strip():
            sections.append(
                Section(
                    heading_path=[title for _, title in heading_stack],
                    content=current.content.strip(),
                )
            )

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        current = Section()

    if current.content.strip():
        sections.append(
            Section(
                heading_path=[title for _, title in heading_stack],
                content=current.content.strip(),
            )
        )

    return sections


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text exceeding ``max_chars`` on paragraph boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split(PARAGRAPH_SEPARATOR)
    parts: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        candidate = paragraph if not buffer else buffer + PARAGRAPH_SEPARATOR + paragraph

        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            parts.append(buffer)
            tail = buffer[-overlap_chars:] if overlap_chars else ""
            buffer = (tail + PARAGRAPH_SEPARATOR + paragraph) if tail else paragraph
        else:
            buffer = paragraph

        while len(buffer) > max_chars:
            parts.append(buffer[:max_chars])
            buffer = buffer[max_chars - overlap_chars :] if overlap_chars else ""

    if buffer.strip():
        parts.append(buffer)

    return parts


def _merge_undersized(parts: list[str], min_chars: int) -> list[str]:
    """Merge fragments below ``min_chars`` into the preceding part."""
    if not parts:
        return []

    merged: list[str] = [parts[0]]
    for part in parts[1:]:
        if len(part) < min_chars:
            merged[-1] = merged[-1] + PARAGRAPH_SEPARATOR + part
        else:
            merged.append(part)
    return merged


def chunk_document(
    document: Document,
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    max_heading_depth: int,
) -> list[Chunk]:
    """Split a document into heading-aware, size-bounded chunks."""
    sections = split_into_sections(document.content, max_depth=max_heading_depth)
    if not sections:
        sections = [Section(heading_path=[], content=document.content.strip())]

    chunks: list[Chunk] = []
    index = 0

    for section in sections:
        parts = _split_oversized(section.content, max_chars, overlap_chars)
        parts = _merge_undersized(parts, min_chars)

        for part in parts:
            text = part.strip()
            if len(text) < min_chars:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{index}",
                    doc_id=document.doc_id,
                    relative_id=document.relative_id,
                    source_path=document.source_path,
                    doc_title=document.title,
                    heading_path=list(section.heading_path),
                    content=text,
                    char_count=len(text),
                    chunk_index=index,
                    section_path=list(document.section_path),
                    source_name=document.source_name,
                    language=document.language,
                )
            )
            index += 1

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    max_heading_depth: int,
) -> list[Chunk]:
    """Chunk every document and return the combined list."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
                max_heading_depth=max_heading_depth,
            )
        )

    logger.info(
        "documents_chunked",
        document_count=len(documents),
        chunk_count=len(chunks),
        mean_chunk_chars=(sum(c.char_count for c in chunks) // len(chunks) if chunks else 0),
    )
    return chunks
