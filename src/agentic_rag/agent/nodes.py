"""Node implementations for the agentic retrieval graph.

Each node is a plain function of ``(state, deps)`` returning only the state
keys it changed. Keeping them free of graph machinery means every node can be
tested on its own with a scripted provider, and the graph module is left with
nothing but wiring.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.agent.prompts import (
    ABSTENTION_MESSAGE,
    ANALYSIS_SYSTEM,
    ANALYSIS_TEMPLATE,
    GENERATION_SYSTEM,
    GENERATION_TEMPLATE,
    GRADING_SYSTEM,
    GRADING_TEMPLATE,
    REWRITE_SYSTEM,
    REWRITE_TEMPLATE,
    UNSUPPORTED_MESSAGE,
    VERIFICATION_SYSTEM,
    VERIFICATION_TEMPLATE,
    format_passages,
)
from agentic_rag.agent.state import (
    AgentState,
    GradedHit,
    QueryType,
    StepRecord,
    Termination,
    Verdict,
)
from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.llm.parsing import ParseError, clamp, parse_json_object
from agentic_rag.llm.provider import ChatProvider, Message
from agentic_rag.obs.logging import get_logger
from agentic_rag.retrieval.hybrid import RetrievalConfig
from agentic_rag.retrieval.store import SearchHit

logger = get_logger(__name__)

CITATION_PATTERN = r"\[(\d+)\]"


@dataclass
class GraphDeps:
    """Everything the nodes need from the outside world."""

    provider: ChatProvider
    retriever: Any
    settings: Settings = field(default_factory=get_settings)
    retrieval_config: RetrievalConfig = field(
        default_factory=lambda: RetrievalConfig("agent_default", use_bm25=True, use_reranker=True)
    )


def _timed(node: str, iteration: int, start: float, **detail: Any) -> StepRecord:
    """Return a step record for a node that started at ``start``."""
    return StepRecord(
        node=node,
        iteration=iteration,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        detail=detail,
    )


def analyse_query(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Classify the question and prepare an initial search query.

    A failure here is not fatal: the raw question is a serviceable search
    query, so a parse error degrades to the identity transform rather than
    ending the request.
    """
    start = time.perf_counter()
    question = state["question"]

    response = deps.provider.complete(
        [Message("user", ANALYSIS_TEMPLATE.format(question=question))],
        system=ANALYSIS_SYSTEM,
    )

    try:
        parsed = parse_json_object(response.text)
        raw_type = str(parsed.get("query_type", "factual"))
        query_type = QueryType(raw_type) if raw_type in set(QueryType) else QueryType.FACTUAL
        search_query = str(parsed.get("search_query") or question).strip() or question
        notes = str(parsed.get("notes", ""))
    except ParseError as error:
        logger.warning("analysis_parse_failed", error=str(error))
        query_type, search_query, notes = QueryType.FACTUAL, question, ""

    return {
        "query_type": query_type,
        "search_query": search_query,
        "analysis_notes": notes,
        "attempted_queries": [search_query],
        "steps": [
            _timed(
                "analyse_query",
                state["rewrite_count"],
                start,
                query_type=str(query_type),
                search_query=search_query,
            )
        ],
    }


def retrieve(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Retrieve candidate passages for the current search query."""
    start = time.perf_counter()

    candidates = deps.retriever.retrieve(
        state["search_query"],
        deps.retrieval_config,
        context=state["tenant"],
        language=state.get("language"),
    )

    return {
        "candidates": candidates,
        "steps": [
            _timed(
                "retrieve",
                state["rewrite_count"],
                start,
                query=state["search_query"],
                candidates=len(candidates),
            )
        ],
    }


def grade_relevance(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Score each candidate passage for relevance to the question.

    Grading is the gate that makes the loop meaningful. Without it the graph
    would answer from whatever retrieval happened to return, and a rewrite
    would never be triggered.
    """
    start = time.perf_counter()
    candidates = state["candidates"]

    if not candidates:
        return {
            "graded": [],
            "relevant": [],
            "mean_relevance": 0.0,
            "steps": [_timed("grade_relevance", state["rewrite_count"], start, graded=0)],
        }

    response = deps.provider.complete(
        [
            Message(
                "user",
                GRADING_TEMPLATE.format(
                    question=state["question"],
                    passages=format_passages([c.content for c in candidates]),
                ),
            )
        ],
        system=GRADING_SYSTEM,
    )

    scores: dict[int, tuple[float, str]] = {}
    try:
        parsed = parse_json_object(response.text)
        for entry in parsed.get("scores", []):
            if not isinstance(entry, dict):
                continue
            position = entry.get("passage")
            if isinstance(position, int) and 1 <= position <= len(candidates):
                scores[position] = (
                    clamp(entry.get("score")),
                    str(entry.get("reason", "")),
                )
    except ParseError as error:
        logger.warning("grading_parse_failed", error=str(error))

    threshold = deps.settings.agent.relevance_threshold
    graded = [
        GradedHit(
            hit=hit, score=scores.get(index, (0.0, ""))[0], reason=scores.get(index, (0.0, ""))[1]
        )
        for index, hit in enumerate(candidates, start=1)
    ]
    graded.sort(key=lambda g: (-g.score, g.chunk_id))
    relevant = [g.hit for g in graded if g.score >= threshold]
    mean = sum(g.score for g in graded) / len(graded) if graded else 0.0

    return {
        "graded": graded,
        "relevant": relevant,
        "mean_relevance": round(mean, 4),
        "steps": [
            _timed(
                "grade_relevance",
                state["rewrite_count"],
                start,
                graded=len(graded),
                relevant=len(relevant),
                mean_relevance=round(mean, 4),
            )
        ],
    }


def rewrite_query(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Reformulate the search query after a failed relevance check."""
    start = time.perf_counter()
    attempted = state["attempted_queries"]

    reason = (
        "no passages were retrieved"
        if not state["candidates"]
        else f"mean relevance was {state['mean_relevance']:.2f}, below threshold"
    )

    response = deps.provider.complete(
        [
            Message(
                "user",
                REWRITE_TEMPLATE.format(
                    question=state["question"],
                    attempted="\n".join(f"- {q}" for q in attempted),
                    reason=reason,
                ),
            )
        ],
        system=REWRITE_SYSTEM,
    )

    try:
        parsed = parse_json_object(response.text)
        rewritten = str(parsed.get("search_query", "")).strip()
    except ParseError as error:
        logger.warning("rewrite_parse_failed", error=str(error))
        rewritten = ""

    # A rewrite that repeats an earlier attempt would loop without progress.
    if not rewritten or rewritten in attempted:
        rewritten = state["question"]
        if rewritten in attempted:
            rewritten = f"{state['question']} configuration reference"

    count = state["rewrite_count"] + 1
    logger.info("query_rewritten", attempt=count, query=rewritten)

    return {
        "search_query": rewritten,
        "rewrite_count": count,
        "attempted_queries": [*attempted, rewritten],
        "steps": [_timed("rewrite_query", count, start, query=rewritten, reason=reason)],
    }


def rerank(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Assemble the final context window from the relevant passages.

    The cross-encoder has already run inside retrieval; this node trims the
    surviving passages to the context budget and drops near-duplicates, which
    otherwise consume the window with the same sentence three times over.
    """
    start = time.perf_counter()
    budget = deps.settings.agent.max_context_chunks

    seen_documents: dict[str, int] = {}
    context: list[SearchHit] = []
    for hit in state["relevant"]:
        occurrences = seen_documents.get(hit.doc_id, 0)
        if occurrences >= 3:
            continue
        seen_documents[hit.doc_id] = occurrences + 1
        context.append(hit)
        if len(context) >= budget:
            break

    return {
        "context": context,
        "steps": [
            _timed(
                "rerank",
                state["rewrite_count"],
                start,
                selected=len(context),
                from_relevant=len(state["relevant"]),
            )
        ],
    }


def generate(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Generate an answer grounded in the assembled context."""
    start = time.perf_counter()
    context = state["context"]

    if not context:
        return {
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
            "termination": Termination.NO_RELEVANT_CONTEXT,
            "steps": [_timed("generate", state["rewrite_count"], start, abstained=True)],
        }

    response = deps.provider.complete(
        [
            Message(
                "user",
                GENERATION_TEMPLATE.format(
                    question=state["question"],
                    passages=format_passages([c.content for c in context]),
                ),
            )
        ],
        system=GENERATION_SYSTEM,
    )

    return {
        "answer": response.text.strip(),
        "citations": [hit.citation for hit in context],
        "steps": [
            _timed(
                "generate",
                state["rewrite_count"],
                start,
                context_chunks=len(context),
                output_tokens=response.output_tokens,
            )
        ],
    }


def verify(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    """Check the answer against its context and suppress it if unsupported.

    This is the last gate before an answer reaches a user. Generation is
    grounded by prompt; verification is grounded by measurement, and only the
    second one is a control.
    """
    start = time.perf_counter()

    if state["termination"] == Termination.NO_RELEVANT_CONTEXT:
        return {
            "verdict": Verdict.NOT_RUN,
            "steps": [_timed("verify", state["rewrite_count"], start, skipped=True)],
        }

    if not deps.settings.agent.enable_verification:
        return {
            "verdict": Verdict.NOT_RUN,
            "termination": Termination.ANSWERED,
            "steps": [_timed("verify", state["rewrite_count"], start, disabled=True)],
        }

    response = deps.provider.complete(
        [
            Message(
                "user",
                VERIFICATION_TEMPLATE.format(
                    question=state["question"],
                    passages=format_passages([c.content for c in state["context"]]),
                    answer=state["answer"],
                ),
            )
        ],
        system=VERIFICATION_SYSTEM,
    )

    try:
        parsed = parse_json_object(response.text)
        faithfulness = clamp(parsed.get("faithfulness"), default=1.0)
        notes = str(parsed.get("notes", ""))
    except ParseError as error:
        # Failing open here would make the control decorative. A verification
        # pass that cannot be read is a verification pass that did not happen.
        logger.warning("verification_parse_failed", error=str(error))
        faithfulness, notes = 0.0, "verification response could not be parsed"

    threshold = deps.settings.agent.faithfulness_threshold
    if faithfulness >= threshold:
        verdict = Verdict.SUPPORTED
    elif faithfulness >= threshold / 2:
        verdict = Verdict.PARTIAL
    else:
        verdict = Verdict.UNSUPPORTED

    suppressed = verdict == Verdict.UNSUPPORTED
    updates: dict[str, Any] = {
        "verdict": verdict,
        "faithfulness": faithfulness,
        "verification_notes": notes,
        "suppressed": suppressed,
        "termination": (
            Termination.UNSUPPORTED_ANSWER_BLOCKED if suppressed else Termination.ANSWERED
        ),
        "steps": [
            _timed(
                "verify",
                state["rewrite_count"],
                start,
                verdict=str(verdict),
                faithfulness=faithfulness,
            )
        ],
    }
    if suppressed:
        logger.warning(
            "answer_suppressed",
            faithfulness=faithfulness,
            request_id=state["request_id"],
        )
        updates["answer"] = UNSUPPORTED_MESSAGE
        updates["citations"] = []
    return updates


def should_rewrite(state: AgentState, settings: Settings | None = None) -> str:
    """Decide whether to reformulate the query or proceed to reranking.

    This is the conditional edge that turns a pipeline into an agent. It is
    also where an unbounded loop would live, so the rewrite budget is checked
    here rather than being left to the nodes.
    """
    settings = settings or get_settings()
    enough = len(state["relevant"]) >= settings.agent.min_relevant_chunks

    if enough:
        return "rerank"
    if state["rewrite_count"] >= settings.agent.max_rewrites:
        return "give_up"
    return "rewrite_query"


def exhausted_state(state: AgentState) -> dict[str, Any]:
    """Terminal update applied when the rewrite budget runs out."""
    return {
        "answer": ABSTENTION_MESSAGE,
        "citations": [],
        "context": [],
        "termination": Termination.REWRITE_BUDGET_EXHAUSTED,
        "steps": [
            StepRecord(
                node="give_up",
                iteration=state["rewrite_count"],
                duration_ms=0.0,
                detail={"attempts": len(state["attempted_queries"])},
            )
        ],
    }


NODE_FUNCTIONS: dict[str, Callable[[AgentState, GraphDeps], dict[str, Any]]] = {
    "analyse_query": analyse_query,
    "retrieve": retrieve,
    "grade_relevance": grade_relevance,
    "rewrite_query": rewrite_query,
    "rerank": rerank,
    "generate": generate,
    "verify": verify,
}
