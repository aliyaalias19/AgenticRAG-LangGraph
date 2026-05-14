"""
Core domain models — shared across the entire pipeline.

Pydantic models instead of plain dataclasses:
- Automatic validation at construction
- JSON serialisation built-in (.model_dump(), .model_json())
- OpenAPI schema generation for free
- IDE type-checking support
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A specific passage from a source chunk that supports a claim in the answer."""

    chunk_id: int = Field(..., description="1-based index into retrieved chunks")
    page: int = Field(..., description="Source document page number")
    source_file: str = Field(..., description="Originating filename")
    snippet: str = Field(..., description="Sentence(s) from the chunk that back the claim")
    full_chunk: str = Field(..., description="Full chunk text for context panel in UI")
    similarity_score: Optional[float] = Field(
        None, description="Reranker score (0-1), None if reranking was skipped"
    )


class RetrievedChunk(BaseModel):
    """A document chunk with retrieval scores attached for full observability."""

    content: str
    metadata: dict = Field(default_factory=dict)

    # Retrieval scores — populated by the retrieval pipeline
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None       # Reciprocal Rank Fusion score
    rerank_score: Optional[float] = None    # Cross-encoder score

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "unknown")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0)) + 1


class RetrievalMetadata(BaseModel):
    """Diagnostic metadata produced by the retrieval pipeline.

    Why expose this: during an interview (and in production observability tooling),
    being able to trace exactly how chunks were retrieved — which searcher found
    them, what scores they had — is critical for debugging recall issues.
    """

    query_used: str = Field(..., description="Actual query used (may be rewritten)")
    vector_results_count: int
    bm25_results_count: int
    fused_results_count: int
    reranking_applied: bool
    retrieval_latency_ms: float
