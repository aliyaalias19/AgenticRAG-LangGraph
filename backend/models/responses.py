"""API response models.

Richer than the original: adds confidence_score, hallucination_score,
retrieval_metadata, and latency_ms so the UI can surface quality signals.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .domain import Citation, RetrievalMetadata


class CitationResponse(BaseModel):
    chunk_id: int
    page: int
    source_file: str
    snippet: str
    full_chunk: str
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]

    # Query processing
    original_query: str
    rewritten_query: str

    # Two raw quality signals. NO composite — the previous
    # confidence_score combined them with uncalibrated weights and is gone.
    relevance_score: int = Field(
        ...,
        description="1-10: how well retrieved chunks matched the query"
    )
    hallucination_score: float = Field(
        ...,
        description="0-1: estimated fraction of answer claims not grounded in context"
    )

    # Execution trace
    steps_taken: int
    reasoning_trace: List[str]

    # Performance
    latency_ms: float

    # Retrieval diagnostics
    retrieval_metadata: Optional[RetrievalMetadata] = None


class IngestResponse(BaseModel):
    status: str
    file: str
    chunks: int
    embedding_provider: str
    message: Optional[str] = None


class DocumentsResponse(BaseModel):
    documents: List[str]
    count: int


class DeleteResponse(BaseModel):
    status: str
    file: str
    chunks_deleted: int


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_connected: bool
    ollama_models: List[str]
    ingested_files: List[str]
    features: List[str]


class EvalMetrics(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class EvalThresholdFailure(BaseModel):
    metric: str
    value: Optional[float] = None
    threshold: float


class EvalResponse(BaseModel):
    status: str
    metrics: EvalMetrics
    total_questions: int
    per_question: List[dict]
    interpretation: dict
    # Declared shipping bar — see backend/evaluation/ragas_eval.py.
    # passed=False means at least one metric fell below the threshold;
    # CI should gate the deploy on this flag.
    passed: bool = True
    threshold_failures: List[EvalThresholdFailure] = Field(default_factory=list)
