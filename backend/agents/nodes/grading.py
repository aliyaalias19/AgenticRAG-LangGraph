"""
Grading Node — scores retrieved chunks for relevance to the query.

Two-tier approach (cost optimisation):
  1. Heuristic (0 tokens): keyword overlap score.
     - ≥ 7/10: accept, skip LLM call entirely (~70% of queries)
     - < 3/10: reject immediately, signal rewrite
     - 3–6:   borderline → escalate to Haiku for nuanced scoring
  2. Haiku LLM grader: used only on borderline scores to avoid the cost
     of running an LLM call on clearly-good or clearly-bad retrievals.

This two-tier design reduces grading cost by ~70% vs always calling an LLM
while maintaining accuracy on the hard cases where heuristics disagree.

Token cost: 0 for 70% of queries, ~200 input tokens (Haiku) for 30%.
"""

import json
import logging
import re
import time

from langchain_core.documents import Document

from agents.state import RAGState
from core.config import get_settings
from core.exceptions import GradingTimeout
from infrastructure.circuit_breaker import CircuitBreaker
from services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

_llm_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

# Grading prompt — generous, semantic, JSON-only.
#
# The original prompt scored chunks against literal answer-presence, which
# under-scored chunks that ANSWER A QUESTION BY IMPLICATION — e.g. "eligible
# if 18 and above" implicitly answers "what's the minimum age". The generator
# routinely produced correct grounded answers from such chunks while the
# grader returned 2/10, triggering pointless rewrite loops. This rewrite
# explicitly tells the grader that implied / definitional / eligibility-style
# matches count, and pins the scoring rubric to four discrete anchors so the
# model has less room to drift.
_GRADE_PROMPT = """\
You are grading whether retrieved chunks let a careful reader answer a query.

Score 1-10 using these anchors:
  10 — chunks contain the answer directly and unambiguously
  7  — chunks contain enough information to derive the answer with one
        small inference (e.g. "eligible if 18+" answers "minimum age")
  5  — chunks are on-topic and partially informative but miss the
        specific fact asked for
  3  — chunks are tangentially related (same document section, same
        general topic) but do not enable an answer
  1  — chunks are off-topic

Be generous on implied answers — definitions, eligibility criteria,
rules, and surrounding context that lets a reader infer the answer
score 6-8, not 2-3.

Query: {query}

Chunks:
{chunks}

Return ONLY a JSON object — no prose, no markdown fences:
{{"score": <1-10>, "feedback": "<<= 10 words>"}}"""

_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "for", "and", "or",
    "are", "how", "does", "do", "was", "were", "be", "been", "have",
    "has", "had", "will", "would", "could", "should", "may", "might",
    "to", "at", "by", "with", "from", "that", "this",
}


def _heuristic_score(query: str, chunks: list) -> int:
    """Zero-cost keyword overlap score (no LLM)."""
    if not chunks:
        return 0
    query_words = {
        w.lower().strip("?.,") for w in query.split()
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not query_words:
        return 5

    total = 0.0
    for chunk in chunks:
        chunk_words = set(chunk.page_content.lower().split())
        overlap = len(query_words & chunk_words) / len(query_words)
        total += overlap

    return min(10, int((total / len(chunks)) * 15))


def _llm_grade(query: str, chunks: list) -> tuple[int, str]:
    """LLM-based grading via the fast tier. Returns (score, feedback).

    JSON-mode output:
      Ollama supports a ``format="json"`` parameter that forces the model
      to emit valid JSON instead of conversational text. We bind that
      flag per-call so the structured-output guarantee is local to the
      grader and doesn't affect the rest of the fast-tier usage. For
      Bedrock/OpenAI, ``.bind(format=...)`` is silently ignored at the
      LangChain layer, so the same code path works across providers.
    """
    grade_llm = get_llm(tier="fast")
    chunk_text = "\n\n---\n\n".join(c.page_content[:300] for c in chunks)
    prompt = _GRADE_PROMPT.format(query=query, chunks=chunk_text)

    try:
        # Use the structured-output flag where the provider supports it.
        # For Ollama this is a guarantee; other providers ignore it.
        try:
            bound = grade_llm.bind(format="json")
        except Exception:
            bound = grade_llm  # provider doesn't support bind(format=...)
        response = _llm_breaker.call(bound.invoke, prompt)
    except Exception as exc:
        raise GradingTimeout(f"Grading LLM call failed: {exc}") from exc

    raw = re.sub(r"```json|```", "", response.content.strip()).strip()
    try:
        parsed = json.loads(raw)
        score = max(1, min(10, int(parsed.get("score", 5))))
        feedback = parsed.get("feedback", "")
    except Exception:
        match = re.search(r"\b(10|[1-9])\b", raw)
        score = int(match.group(1)) if match else 5
        feedback = "Parse error — used regex fallback"

    return score, feedback


def grade_node(state: RAGState) -> dict:
    """Score retrieved chunks and decide whether to rewrite or answer."""
    query = state.get("rewritten_query") or state["query"]
    chunks = state.get("retrieved_chunks", [])

    heuristic = _heuristic_score(query, chunks)

    if heuristic >= 7:
        score, feedback = heuristic, "Strong keyword match (heuristic)"
        logger.debug("Grade: %d/10 heuristic fast-pass", score)
    elif heuristic < 3:
        score, feedback = heuristic, "Poor keyword match — rewrite recommended"
        logger.debug("Grade: %d/10 heuristic fast-reject", score)
    else:
        try:
            score, feedback = _llm_grade(query, chunks)
            logger.debug("Grade: %d/10 LLM — %s", score, feedback)
        except GradingTimeout as exc:
            logger.warning("Grading timeout, using heuristic: %s", exc)
            score, feedback = heuristic, f"Heuristic fallback (timeout): {exc}"

    return {
        "relevance_score": score,
        "retrieval_feedback": feedback,
        "reasoning_trace": [f"[Grade] {score}/10 — {feedback}"],
    }
