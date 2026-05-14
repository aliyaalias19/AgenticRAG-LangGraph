"""
Hallucination Check Node — post-generation grounding verification.

What "hallucination" means in RAG:
  A hallucination is a claim in the answer that is NOT supported by any
  retrieved chunk. This is distinct from relevance grading (which evaluates
  chunks against the question) — hallucination checks evaluate the answer
  against the chunks.

Why check after generation:
  Even with relevant context, LLMs occasionally generate plausible-sounding
  but unsupported claims. Detecting these post-generation allows the system
  to surface a confidence signal to the user ("this answer may contain
  unsupported claims") without blocking the response.

Heuristic fast-path:
  If relevance_score >= 8, the retrieval quality was excellent, so the
  probability of hallucination is low. We skip the LLM call to save tokens.
  The threshold is a tradeoff — lower it to catch more edge cases, raise it
  to save more tokens.

Two raw signals, no composite:
  Earlier versions returned a single ``confidence_score`` computed as
  ``(relevance/10)*0.6 + (1-h_score)*0.4``. The 0.6/0.4 weights were
  uncalibrated — they had no derivation against labelled correctness data.
  Returning a single number that customers might treat as ground truth is
  worse than returning the two raw signals separately and letting them be
  combined with calibration done by whoever has the labels.

  This node now stops writing ``confidence_score`` to state. The API
  surfaces ``relevance_score`` (1-10, from grading) and
  ``hallucination_score`` (0-1, from this node) as independent fields.

Token cost: 0 for high-relevance queries, ~300 input + ~20 output (Haiku) otherwise.
"""

import json
import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document

from agents.state import RAGState
from core.config import get_settings
from services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

_HALLUCINATION_PROMPT = """\
You are a fact-checker. For each factual claim in the answer, verify whether it is supported by the context.
Return a score from 0.0 (fully grounded) to 1.0 (heavily hallucinated).

Answer: {answer}

Context (excerpts):
{context}

Return JSON only: {{"hallucination_score": <0.0-1.0>, "verdict": "grounded|partial|hallucinated"}}"""


def _llm_hallucination_check(answer: str, chunks: List[Document]) -> Tuple[float, str]:
    """Ask Haiku to verify answer grounding. Returns (score, verdict)."""
    context_summary = "\n\n".join(
        f"[{i+1}] {c.page_content[:200]}" for i, c in enumerate(chunks)
    )
    prompt = _HALLUCINATION_PROMPT.format(answer=answer[:800], context=context_summary)

    try:
        llm = get_llm(tier="fast")
        response = llm.invoke(prompt)
        raw = re.sub(r"```json|```", "", response.content.strip()).strip()
        parsed = json.loads(raw)
        score = max(0.0, min(1.0, float(parsed.get("hallucination_score", 0.3))))
        verdict = parsed.get("verdict", "unknown")
        return score, verdict
    except Exception as exc:
        logger.warning("Hallucination check failed (%s) — defaulting to 0.3", exc)
        return 0.3, "check_failed"


def hallucination_check_node(state: RAGState) -> dict:
    """Verify answer grounding. Writes only the raw hallucination_score —
    no composite confidence is fabricated."""
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])
    relevance_score = state.get("relevance_score", 0)

    if not answer or not chunks:
        return {
            "hallucination_score": 0.5,
            "reasoning_trace": ["[HallucinationCheck] Skipped — no answer/chunks"],
        }

    # High relevance → skip expensive LLM check (fast path)
    if relevance_score >= 8:
        h_score = 0.05
        logger.debug("HallucinationCheck fast-path", extra={"relevance": relevance_score})
        return {
            "hallucination_score": h_score,
            "reasoning_trace": [
                f"[HallucinationCheck] Fast-path (relevance={relevance_score}/10) "
                f"· h_score={h_score:.2f}"
            ],
        }

    h_score, verdict = _llm_hallucination_check(answer, chunks)
    logger.info("HallucinationCheck", extra={"h_score": h_score, "verdict": verdict})

    return {
        "hallucination_score": h_score,
        "reasoning_trace": [
            f"[HallucinationCheck] h_score={h_score:.2f} ({verdict})"
        ],
    }
