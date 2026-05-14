"""
Query Rewriting Node — reformulates the query when retrieval quality is low.

Three strategies, chosen based on rewrite history:
  precise:    synonym substitution, more specific terminology
  broadened:  remove over-specific constraints that are limiting recall
  decomposed: break a compound question into a single focused sub-question

Why rewriting improves recall:
  The user's original phrasing may not match the document vocabulary.
  "What are the fees?" may miss chunks that say "charges", "costs", or
  "expenses". A rewriter trained on instruction-following can bridge this
  vocabulary gap. This is the core value of the agentic loop over one-shot RAG.

Token cost: ~150 input + ~30 output (Haiku).
"""

import logging

from agents.state import RAGState
from core.config import get_settings
from services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

_REWRITE_PROMPT = """\
Rewrite the query using the {strategy} strategy to improve document retrieval.
precise=use synonyms/specific terms | broadened=remove over-specific filters | decomposed=single focused sub-question
Current relevance score: {score}/10. Feedback: {feedback}
Original query: {query}
Output the rewritten query only (no explanation)."""


def rewrite_node(state: RAGState) -> dict:
    """Rewrite the current query using the most appropriate strategy."""
    rewrite_count = state.get("rewrite_count", 0)
    query = state.get("rewritten_query") or state["query"]
    score = state.get("relevance_score", 0)
    feedback = state.get("retrieval_feedback", "")

    # Escalate strategy on repeated rewrites
    if rewrite_count == 0:
        strategy = "precise"
    elif rewrite_count == 1:
        strategy = "broadened"
    else:
        strategy = "decomposed"

    llm = get_llm(tier="rewrite")
    prompt = _REWRITE_PROMPT.format(
        strategy=strategy,
        score=score,
        feedback=feedback,
        query=query,
    )

    try:
        response = llm.invoke(prompt)
        new_query = response.content.strip()
        if not (5 <= len(new_query) <= 300):
            new_query = query  # fallback: keep original if output is clearly wrong
    except Exception as exc:
        logger.warning("Rewrite failed (%s), keeping original query", exc)
        new_query = query

    logger.info("Rewrite[%d] strategy=%s: %r → %r", rewrite_count, strategy, query[:60], new_query[:60])

    return {
        "rewritten_query": new_query,
        "rewrite_count": rewrite_count + 1,
        "reasoning_trace": [
            f"[Rewrite#{rewrite_count+1}] strategy={strategy} → {new_query!r}"
        ],
    }
