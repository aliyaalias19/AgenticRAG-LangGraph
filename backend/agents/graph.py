"""
LangGraph Agentic RAG state graph.

Is LangGraph worth the dependency here? Honest answer:
──────────────────────────────────────────────────────
  For *this* size of graph (six nodes, mostly linear, one conditional
  edge), LangGraph is overhead. A 60-line plain-Python state machine
  with a typed dict and a dispatch dict would do the job with one fewer
  framework to pin. I do not pretend otherwise.

  The reasons it stays:

  1. Typed state with append semantics for ``reasoning_trace`` via
     ``Annotated[List[str], operator.add]``. Hand-rolling that contract
     is straightforward but easy to get wrong (every node must remember
     to merge instead of replace).

  2. Nodes are pure functions ``state -> dict``. They unit-test cleanly
     without any framework setup. ``backend/tests/test_nodes.py`` runs
     43 of them in milliseconds because LangGraph never has to be
     imported to exercise the logic.

  3. Migration cost to interrupts (human-in-the-loop on low-confidence
     answers), checkpointers (resume after crash), and parallel fan-out
     (multi-document compare in a single graph) is near zero — those
     features are first-class in LangGraph and would be real work in a
     hand-rolled state machine.

  If the system never grows into branching beyond the one rewrite loop,
  I would replace LangGraph with ~60 lines of plain Python. Today the
  framework earns its keep through (1) and (2). The day it stops, I rip
  it out — that decision is documented here so a future reader knows
  the tradeoff was considered, not inherited by accident.

Replaces the previous imperative while-loop:
────────────────────────────────────────────
  Old approach:
    while steps < max:
        decision = _decide(state)   # LLM call for routing — expensive
        if decision == "retrieve": ...
        elif decision == "grade": ...

  New approach:
    Graph topology defined ONCE. Each node is a pure function.
    Conditional edges express routing logic in code, not in prompts.
    Zero LLM tokens spent on routing decisions.

Graph topology:
                   ┌────────────────────┐
                   │   analyze_query    │ (classify tier, set budget)
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │      retrieve      │ (BM25 + vector + RRF + rerank)
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │       grade        │ (heuristic + optional Haiku)
                   └─────────┬──────────┘
                             │
                  ┌──────────▼───────────┐
                  │  route_after_grade   │ ← conditional edge
                  └──────┬───────┬───────┘
                  score≥4 │       │ score<4 & steps<max
                         │       │
               ┌─────────▼──┐ ┌──▼──────────┐
               │  generate  │ │   rewrite   │
               └─────────┬──┘ └──┬──────────┘
                         │       │ (→ retrieve)
               ┌─────────▼───────┘
               │  hallucination_check  │
               └─────────────────────┘
                         │
                        END
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List

from langgraph.graph import END, StateGraph

from agents.nodes import (
    generate_node,
    grade_node,
    hallucination_check_node,
    query_analysis_node,
    retrieve_node,
    rewrite_node,
)
from agents.state import RAGState, create_initial_state
from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing function (replaces the original _decide() LLM call entirely)
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_grade(state: RAGState) -> str:
    """Conditional edge: decide whether to answer or rewrite-then-retrieve.

    This replaces the expensive Haiku-based _decide() from the original code.
    All routing is now deterministic — zero LLM tokens for routing decisions.

    Why this is better:
      The original _decide() used an LLM to choose the next tool, but 95% of
      the time the deterministic guards overrode it anyway. Removing the LLM
      call saves ~200 tokens per step and eliminates a failure mode where
      malformed JSON from the LLM caused incorrect routing.
    """
    score = state.get("relevance_score", 0)
    steps = state.get("steps_taken", 0)
    rewrites = state.get("rewrite_count", 0)
    max_steps = state.get("effective_max_steps", 5)

    settings = get_settings()

    if score >= settings.min_relevance_score:
        logger.debug("Route → generate (score=%d >= threshold=%d)", score, settings.min_relevance_score)
        return "generate"

    if steps >= max_steps or rewrites >= settings.max_rewrites:
        logger.debug("Route → generate (best-effort: steps=%d rewrites=%d)", steps, rewrites)
        return "generate"

    logger.debug("Route → rewrite (score=%d, rewrite#%d)", score, rewrites + 1)
    return "rewrite"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph():
    """Construct and compile the LangGraph RAG pipeline.

    Returns a compiled graph that accepts RAGState and produces RAGState.
    The compiled graph is immutable — build it once at startup and reuse.
    """
    workflow = StateGraph(RAGState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("analyze_query",       query_analysis_node)
    workflow.add_node("retrieve",            retrieve_node)
    workflow.add_node("grade",               grade_node)
    workflow.add_node("rewrite",             rewrite_node)
    workflow.add_node("generate",            generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("analyze_query")

    # ── Deterministic edges ───────────────────────────────────────────────────
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve",      "grade")
    workflow.add_edge("rewrite",       "retrieve")   # rewrites loop back to retrieval
    workflow.add_edge("generate",      "hallucination_check")
    workflow.add_edge("hallucination_check", END)

    # ── Conditional edge: after grading ───────────────────────────────────────
    workflow.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate": "generate",
            "rewrite":  "rewrite",
        },
    )

    return workflow.compile()


# Process-level singleton — compiled once at startup
_graph = None


def get_rag_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
        logger.info("LangGraph RAG pipeline compiled")
    return _graph


# ─────────────────────────────────────────────────────────────────────────────
# Public run interfaces
# ─────────────────────────────────────────────────────────────────────────────

def run_rag(
    query: str,
    chat_history: List[dict] | None = None,
    tenant_id: str | None = None,
) -> RAGState:
    """Execute the full agentic RAG pipeline and return the final state.

    This is the non-streaming entry point used by the /query endpoint.
    ``tenant_id`` partitions retrieval — every chunk read during this
    request must belong to the named tenant.
    """
    t0 = time.time()
    graph = get_rag_graph()
    initial = create_initial_state(query, chat_history, tenant_id)
    final: RAGState = graph.invoke(initial)
    latency = (time.time() - t0) * 1000
    final["latency_ms"] = round(latency, 1)  # type: ignore[typeddict-unknown-key]

    logger.info(
        "RAG complete: steps=%d relevance=%d/10 h_score=%.2f citations=%d latency=%.0fms",
        final.get("steps_taken", 0),
        final.get("relevance_score", 0),
        final.get("hallucination_score", 0.0),
        len(final.get("citations", [])),
        latency,
    )
    return final


def run_rag_streaming(query: str, tenant_id: str | None = None) -> Generator[str, None, None]:
    """Stream the RAG pipeline output as newline-delimited JSON events.

    Event sequence (live, in order of arrival):
      {"type": "step",     "node": ..., "message": ...}  — per-node progress
      {"type": "token",    "content": "..."}              — answer tokens
      {"type": "citations","data": [...]}                  — cited sources
      {"type": "meta",     ...}                           — quality metrics

    Why ``graph.stream()`` and not ``graph.invoke()``:
      The previous implementation called ``invoke()`` which blocks until
      the entire pipeline completes (~30 s for a cold query) before
      emitting a single byte. The user saw a blank screen for the whole
      duration and assumed the system was hung. ``stream(mode="updates")``
      yields each node's delta as soon as it finishes — we forward those
      as ``step`` events so the UI can render a live "thinking" trace,
      matching the UX of Claude / ChatGPT / Cursor and giving users
      meaningful feedback about which stage the graph is in.

    State accumulation:
      ``stream_mode="updates"`` yields only the *delta* per node, not the
      full state. ``reasoning_trace`` has append semantics
      (``Annotated[List[str], operator.add]``) so we merge its deltas;
      every other field is a straight overwrite.
    """
    import time

    t0 = time.time()
    graph = get_rag_graph()
    initial = create_initial_state(query, tenant_id=tenant_id)

    # Local mirror of state — we accumulate the final state ourselves
    # since stream_mode="updates" only yields deltas. The mirror feeds
    # the token / citations / meta phase below.
    state: dict = dict(initial)

    for update in graph.stream(initial, stream_mode="updates"):
        # update has the shape {node_name: {field: value, ...}}
        for node_name, delta in update.items():
            for key, value in delta.items():
                if key == "reasoning_trace":
                    state["reasoning_trace"] = (
                        list(state.get("reasoning_trace") or []) + list(value or [])
                    )
                    for line in (value or []):
                        yield json.dumps({
                            "type":    "step",
                            "node":    node_name,
                            "message": line,
                        }) + "\n"
                else:
                    state[key] = value

    # Emit the pre-generated answer word by word so the client gets a
    # progressive display. NOT a second LLM call — the answer was
    # produced by the generate_node during the stream above.
    full_answer = state.get("answer", "") or ""
    for word in full_answer.split():
        yield json.dumps({"type": "token", "content": word + " "}) + "\n"

    citations = state.get("citations", []) or []
    yield json.dumps({"type": "citations", "data": citations}) + "\n"

    latency = (time.time() - t0) * 1000
    yield json.dumps({
        "type":                "meta",
        "original_query":      query,
        "rewritten_query":     state.get("rewritten_query", query),
        "relevance_score":     state.get("relevance_score", 0),
        "hallucination_score": state.get("hallucination_score", 0.0),
        "steps_taken":         state.get("steps_taken", 0),
        "latency_ms":          round(latency, 1),
    }) + "\n"
