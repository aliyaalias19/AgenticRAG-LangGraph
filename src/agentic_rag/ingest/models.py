"""Typed models for corpus documents and manifests."""

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single source document with its metadata and content."""

    doc_id: str = Field(description="Stable identifier derived from source path")
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

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return the SHA-256 hex digest of the given content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class CorpusManifest(BaseModel):
    """Provenance record describing exactly how a corpus was built."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    repo_url: str
    repo_ref: str
    commit_sha: str = Field(description="Exact commit the corpus was built from")
    docs_subpath: str
    document_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    min_document_chars: int = Field(ge=0)
    corpus_hash: str = Field(description="SHA-256 over all sorted document hashes")

    @staticmethod
    def compute_corpus_hash(documents: list[Document]) -> str:
        """Return a hash covering the full document set, order-independent."""
        digest = hashlib.sha256()
        for doc_hash in sorted(d.content_hash for d in documents):
            digest.update(doc_hash.encode("utf-8"))
        return digest.hexdigest()
