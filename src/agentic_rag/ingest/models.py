"""Typed models for corpus documents and manifests."""

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single source document with its metadata and content."""

    doc_id: str = Field(description="Stable identifier derived from source path")
    relative_id: str = Field(
        default="",
        description="Path-based identifier shared across language variants",
    )
    source_path: str = Field(description="Path relative to the docs root")
    title: str = Field(description="Document title from frontmatter or filename")
    content: str = Field(description="Markdown body with frontmatter removed")
    content_hash: str = Field(description="SHA-256 of the content")
    char_count: int = Field(ge=0)
    section_path: list[str] = Field(
        default_factory=list,
        description="Directory hierarchy, e.g. ['concepts', 'workloads']",
    )
    frontmatter: dict[str, str] = Field(default_factory=dict)
    source_name: str = Field(default="", description="Identifier of the source repo")
    language: str = Field(default="en", description="ISO 639-1 language code")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return the SHA-256 hex digest of the given content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SourceRecord(BaseModel):
    """Provenance for a single ingested source repository."""

    name: str
    repo_url: str
    repo_ref: str
    commit_sha: str
    docs_subpath: str
    language: str
    document_count: int = Field(ge=0)


class CorpusManifest(BaseModel):
    """Provenance record describing exactly how a corpus was built."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sources: list[SourceRecord] = Field(default_factory=list)
    document_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    min_document_chars: int = Field(ge=0)
    corpus_hash: str = Field(description="SHA-256 over all sorted document hashes")
    chunk_count: int = Field(default=0, ge=0)
    chunk_max_chars: int = Field(default=0, ge=0)
    chunk_overlap_chars: int = Field(default=0, ge=0)

    @staticmethod
    def compute_corpus_hash(documents: list[Document]) -> str:
        """Return a hash covering the full document set, order-independent."""
        digest = hashlib.sha256()
        for doc_hash in sorted(d.content_hash for d in documents):
            digest.update(doc_hash.encode("utf-8"))
        return digest.hexdigest()


class Chunk(BaseModel):
    """A retrievable passage derived from a source document."""

    chunk_id: str = Field(description="Stable identifier: '<doc_id>#<index>'")
    doc_id: str = Field(description="Identifier of the parent document")
    relative_id: str = Field(default="")
    source_path: str = Field(description="Path of the parent document")
    doc_title: str = Field(description="Title of the parent document")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Markdown heading hierarchy above this chunk",
    )
    content: str = Field(description="Chunk text without the context prefix")
    char_count: int = Field(ge=0)
    chunk_index: int = Field(ge=0, description="Position within the parent document")
    section_path: list[str] = Field(default_factory=list)
    source_name: str = Field(default="")
    language: str = Field(default="en")

    @property
    def contextual_text(self) -> str:
        """Return the chunk text prefixed with its document and heading context."""
        parts = [self.doc_title, *self.heading_path]
        return f"{' > '.join(parts)}\n\n{self.content}"
