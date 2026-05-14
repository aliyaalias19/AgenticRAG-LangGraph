"""
LangGraph state definition for the Agentic RAG pipeline.

Why TypedDict instead of a dataclass:
  LangGraph requires the state to be a TypedDict (or a compatible Pydantic
  model). TypedDict gives us static type checking while remaining a plain
  dict at runtime — LangGraph passes state as dict between nodes.

Why Annotated[List[str], operator.add] for reasoning_trace:
  LangGraph's default update behaviour is to REPLACE a field with the value
  returned by a node. For lists we want APPEND semantics: each node adds its
  trace entries without knowing what came before. The operator.add annotation
  tells LangGraph to merge rather than replace.

State lifecycle:
  create_initial_state() → analyze_query → retrieve → grade →
  ┌─ [score < threshold] → rewrite ─┐
  └─ [score >= threshold] ──────────┴→ generate → hallucination_check → END
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    chat_history: List[dict]
    # tenant_id partitions retrieval. The graph never crosses tenants —
    # every retrieval node filters by this value. Defaults to the
    # configured DEFAULT_TENANT in single-tenant demo mode.
    tenant_id: str

    # ── Query analysis ────────────────────────────────────────────────────────
    tier: str                    # "fast" | "smart"
    rewritten_query: str
    effective_max_steps: int

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: List[Any]  # List[langchain_core.documents.Document]
    retrieval_metadata: Dict[str, Any]

    # ── Grading ───────────────────────────────────────────────────────────────
    relevance_score: int         # 1–10
    retrieval_feedback: str

    # ── Loop control ──────────────────────────────────────────────────────────
    steps_taken: int
    rewrite_count: int

    # ── Output ────────────────────────────────────────────────────────────────
    answer: str
    citations: List[dict]

    # ── Quality signals ────────────────────────────────────────────────────────
    # Two independent raw signals. We deliberately do not write a composite
    # confidence_score — see backend/agents/nodes/hallucination.py for the
    # rationale (uncalibrated weights are worse than the two raw numbers).
    hallucination_score: float   # 0=grounded, 1=hallucinated

    # ── Observability ─────────────────────────────────────────────────────────
    # Annotated with operator.add → nodes APPEND rather than REPLACE this list
    reasoning_trace: Annotated[List[str], operator.add]

    # Latency tracking
    latency_start: float         # time.time() at graph entry


def create_initial_state(
    query: str,
    chat_history: List[dict] | None = None,
    tenant_id: str | None = None,
) -> RAGState:
    """Build the initial state dict before the graph starts executing."""
    import time
    from core.config import get_settings
    return RAGState(
        query=query,
        chat_history=chat_history or [],
        tenant_id=tenant_id or get_settings().default_tenant,
        tier="fast",
        rewritten_query=query,
        effective_max_steps=5,
        retrieved_chunks=[],
        retrieval_metadata={},
        relevance_score=0,
        retrieval_feedback="",
        steps_taken=0,
        rewrite_count=0,
        answer="",
        citations=[],
        hallucination_score=0.0,
        reasoning_trace=[],
        latency_start=time.time(),
    )
