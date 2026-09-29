"""Typed state for the agentic retrieval graph.

Every node receives the whole state and returns only the keys it changed.
Making the state explicit and typed is what keeps a cyclic graph debuggable:
when a loop misbehaves, the trace shows which node wrote which field on which
iteration, rather than leaving a tangle of mutated objects.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from agentic_rag.retrieval.store import SearchHit
from agentic_rag.security.tenancy import TenantContext


class QueryType(StrEnum):
    """How the query should be routed."""

    FACTUAL = "factual"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    UNSUPPORTED = "unsupported"


class Verdict(StrEnum):
    """Outcome of post-generation verification."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    NOT_RUN = "not_run"


class Termination(StrEnum):
    """Why the graph stopped."""

    ANSWERED = "answered"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    REWRITE_BUDGET_EXHAUSTED = "rewrite_budget_exhausted"
    UNSUPPORTED_ANSWER_BLOCKED = "unsupported_answer_blocked"
    RUNNING = "running"


@dataclass(frozen=True)
class GradedHit:
    """A retrieved passage with its relevance judgement."""

    hit: SearchHit
    score: float
    reason: str = ""

    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id


@dataclass(frozen=True)
class StepRecord:
    """One node execution, recorded for tracing."""

    node: str
    iteration: int
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def append_steps(left: list[StepRecord], right: list[StepRecord]) -> list[StepRecord]:
    """Reducer that accumulates step records across graph iterations.

    LangGraph replaces a state key with whatever a node returns unless the key
    declares a reducer. The trace must accumulate rather than be overwritten,
    which is exactly the bug that makes cyclic graphs impossible to debug.
    """
    return [*left, *right]


class AgentState(TypedDict, total=False):
    """State threaded through every node of the retrieval graph."""

    # Request
    question: str
    request_id: str
    tenant: TenantContext
    language: str | None

    # Query analysis
    query_type: QueryType
    search_query: str
    analysis_notes: str

    # Retrieval and self-correction
    candidates: list[SearchHit]
    rewrite_count: int
    attempted_queries: list[str]

    # Relevance grading
    graded: list[GradedHit]
    relevant: list[SearchHit]
    mean_relevance: float

    # Reranking and context assembly
    context: list[SearchHit]

    # Generation
    answer: str
    citations: list[str]

    # Verification
    verdict: Verdict
    faithfulness: float
    verification_notes: str
    suppressed: bool

    # Outcome
    termination: Termination
    steps: Annotated[list[StepRecord], append_steps]


def initial_state(
    question: str,
    tenant: TenantContext,
    request_id: str,
    language: str | None = None,
) -> AgentState:
    """Return a fully populated starting state."""
    return AgentState(
        question=question,
        request_id=request_id,
        tenant=tenant,
        language=language,
        query_type=QueryType.FACTUAL,
        search_query=question,
        analysis_notes="",
        candidates=[],
        rewrite_count=0,
        attempted_queries=[],
        graded=[],
        relevant=[],
        mean_relevance=0.0,
        context=[],
        answer="",
        citations=[],
        verdict=Verdict.NOT_RUN,
        faithfulness=0.0,
        verification_notes="",
        suppressed=False,
        termination=Termination.RUNNING,
        steps=[],
    )
