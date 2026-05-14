"""
Domain exception hierarchy.

Using a typed exception hierarchy instead of raising plain Exception:
- Callers can catch specific failure modes and respond appropriately
- HTTP layer maps each exception to the right status code
- Log messages include the exception type for faster triage
"""


class RAGError(Exception):
    """Base class for all pipeline errors."""


class RetrievalError(RAGError):
    """Vector store or BM25 retrieval failed."""


class GradingError(RAGError):
    """LLM grader failed or timed out."""


class GradingTimeout(GradingError):
    """Grading LLM call exceeded the time budget."""


class EmbeddingError(RAGError):
    """Embedding model failed."""


class VectorStoreError(RAGError):
    """ChromaDB operation failed."""


class IngestionError(RAGError):
    """Document ingestion failed."""


class GenerationError(RAGError):
    """Answer generation failed."""


class ConfigurationError(RAGError):
    """Required environment variable missing or invalid."""
