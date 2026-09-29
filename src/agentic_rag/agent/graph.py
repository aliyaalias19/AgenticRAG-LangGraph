"""LangGraph wiring for the agentic retrieval workflow.

    analyse_query -> retrieve -> grade_relevance
                        ^              |
                        |              +-- enough relevant context --> rerank
                        |              |                                 |
                        +-- rewrite ---+ budget remaining             generate
                                       |                                 |
                                       +-- budget exhausted -> give_up  verify
                                                                  |       |
                                                                  +--> END

The cycle between grading and rewriting is the whole point: a pipeline decides
once, an agent decides again with what it learned. The rewrite budget lives on
the conditional edge, so the loop cannot run away no matter how a node behaves.
"""

from dataclasses import dataclass
from typing import Any

from agentic_rag.agent.nodes import (
    GraphDeps,
    analyse_query,
    exhausted_state,
    generate,
    grade_relevance,
    rerank,
    retrieve,
    rewrite_query,
    should_rewrite,
    verify,
)
from agentic_rag.agent.state import AgentState, StepRecord, Termination, initial_state
from agentic_rag.obs.logging import get_logger
from agentic_rag.security.tenancy import TenantContext

logger = get_logger(__name__)

MAX_GRAPH_STEPS = 40


def build_graph(deps: GraphDeps) -> Any:
    """Compile the LangGraph state machine."""
    from langgraph.graph import END, StateGraph

    builder = StateGraph(AgentState)

    builder.add_node("analyse_query", lambda s: analyse_query(s, deps))
    builder.add_node("retrieve", lambda s: retrieve(s, deps))
    builder.add_node("grade_relevance", lambda s: grade_relevance(s, deps))
    builder.add_node("rewrite_query", lambda s: rewrite_query(s, deps))
    builder.add_node("rerank", lambda s: rerank(s, deps))
    builder.add_node("generate", lambda s: generate(s, deps))
    builder.add_node("verify", lambda s: verify(s, deps))
    builder.add_node("give_up", exhausted_state)

    builder.set_entry_point("analyse_query")
    builder.add_edge("analyse_query", "retrieve")
    builder.add_edge("retrieve", "grade_relevance")

    builder.add_conditional_edges(
        "grade_relevance",
        lambda s: should_rewrite(s, deps.settings),
        {
            "rerank": "rerank",
            "rewrite_query": "rewrite_query",
            "give_up": "give_up",
        },
    )

    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("give_up", END)

    return builder.compile()


@dataclass
class AgentRunner:
    """Executes the retrieval graph for one question."""

    deps: GraphDeps

    def __post_init__(self) -> None:
        self._graph = build_graph(self.deps)

    def run(
        self,
        question: str,
        tenant: TenantContext,
        request_id: str,
        language: str | None = None,
    ) -> AgentState:
        """Run the graph to completion and return the final state."""
        state = initial_state(question, tenant, request_id, language)
        final: AgentState = self._graph.invoke(state, config={"recursion_limit": MAX_GRAPH_STEPS})

        logger.info(
            "agent_run_completed",
            request_id=request_id,
            termination=str(final.get("termination")),
            rewrites=final.get("rewrite_count", 0),
            context_chunks=len(final.get("context", [])),
            suppressed=final.get("suppressed", False),
            **tenant.as_log_fields(),
        )
        return final


def trace_summary(state: AgentState) -> list[dict[str, Any]]:
    """Return a compact, serialisable trace of the run."""
    steps: list[StepRecord] = state.get("steps", [])
    return [
        {
            "node": step.node,
            "iteration": step.iteration,
            "duration_ms": step.duration_ms,
            **step.detail,
        }
        for step in steps
    ]


def total_duration_ms(state: AgentState) -> float:
    """Return the summed node duration for a run."""
    return round(sum(step.duration_ms for step in state.get("steps", [])), 2)


def is_answered(state: AgentState) -> bool:
    """Return True when the run produced a usable answer."""
    return state.get("termination") == Termination.ANSWERED
