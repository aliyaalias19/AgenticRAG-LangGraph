"""
Query Analysis Node — first node in the RAG graph.

Responsibilities:
1. Classify query complexity (fast/smart) to set the LLM tier budget.
2. Set the effective step cap (simple queries don't need 5 retrieval steps).

Why separate this into its own node:
  In a production system, query analysis can be extended to detect:
  - Requires live data (route to web search tool)
  - Multi-document comparison (route to synthesis pipeline)
  - PII or restricted topics (route to compliance filter)
  Keeping it isolated means these extensions don't require touching the
  retrieval or generation nodes.

Token cost: 0 (pure heuristic — no LLM call).
"""

import logging

from agents.state import RAGState
from core.config import get_settings
from services.llm import classify_query_complexity

logger = logging.getLogger(__name__)
settings = get_settings()


def query_analysis_node(state: RAGState) -> dict:
    """Classify query complexity and set execution budget."""
    query = state["query"]
    tier = classify_query_complexity(query)

    # Simple queries rarely need more than 3 steps (retrieve → grade → answer).
    # Complex queries get the full budget.
    effective_max = 3 if tier == "fast" else settings.max_steps

    logger.info("QueryAnalysis: tier=%s max_steps=%d query=%r", tier, effective_max, query[:80])

    return {
        "tier": tier,
        "effective_max_steps": effective_max,
        "rewritten_query": query,
        "steps_taken": 0,
        "rewrite_count": 0,
        "reasoning_trace": [
            f"[QueryAnalysis] tier={tier.upper()} · max_steps={effective_max}"
        ],
    }
