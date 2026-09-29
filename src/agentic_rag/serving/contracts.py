"""Typed API contracts.

The request and response shapes are defined once, here, and shared by the
route handlers, the streaming encoder and the tests. Keeping the wire format
independent of the retrieval and model implementations is what lets the
provider or the retrieval policy change without touching a client.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(StrEnum):
    """Server-sent event types emitted during a streamed answer."""

    STATUS = "status"
    TOKEN = "token"  # noqa: S105 — SSE event name, not a credential
    CITATIONS = "citations"
    TRACE = "trace"
    DONE = "done"
    ERROR = "error"


class QueryRequest(BaseModel):
    """A question submitted for answering."""

    question: str = Field(min_length=1, max_length=2000)
    language: str | None = Field(default=None, pattern="^(en|zh)$")
    include_trace: bool = Field(default=False)


class Citation(BaseModel):
    """One cited source passage."""

    chunk_id: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    source_path: str = ""
    language: str = "en"


class QueryResponse(BaseModel):
    """A completed answer with its provenance."""

    request_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_type: str
    termination: str
    verdict: str
    faithfulness: float = 0.0
    suppressed: bool = False
    rewrite_count: int = 0
    duration_ms: float = 0.0
    trace: list[dict[str, Any]] | None = None


class StreamEvent(BaseModel):
    """One server-sent event."""

    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)

    def encode(self) -> str:
        """Return the event in SSE wire format.

        The blank line terminator is not decorative: without it the client
        buffers the event indefinitely, which looks exactly like a hung
        backend.
        """
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


class HealthResponse(BaseModel):
    """Service liveness and dependency status."""

    status: str
    version: str
    provider: str
    collection: str
    chunks_indexed: int | None = None
    lexical_index_loaded: bool = False
