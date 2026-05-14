"""
Generation Node — produces the final grounded answer with inline citations.

Citation format:
  The LLM is instructed to cite sources as [1], [2], etc. after each claim.
  Post-processing then maps each [N] marker to the specific sentence in the
  corresponding chunk that best matches the claim — giving sentence-level
  evidence rather than entire-chunk references.

Why sentence-level citations matter:
  Chunk-level citations (pointing to 800-char chunks) make it hard for
  users to verify claims. Sentence-level citations reduce verification
  time significantly and build trust in the system's answers.

Model selection:
  Simple queries use Haiku (cheap, context-heavy task).
  Complex queries use Sonnet (more reasoning needed).
  This is the ONLY place Sonnet is called — everywhere else uses Haiku.

Token cost: ~1,200 input (context + prompt) + ~200 output (Haiku)
            ~1,500 input + ~400 output (Sonnet for complex queries)
"""

import logging
import re
from typing import Generator, List, Tuple

from langchain_core.documents import Document

from agents.state import RAGState
from core.config import get_settings
from models.domain import Citation
from services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

_ANSWER_PROMPT = """\
Using ONLY the provided context, answer the question concisely.
Cite sources as [1], [2], etc. directly after each factual claim.
If the context is insufficient, say so clearly.

Question: {query}

Context:
{context}

Answer:"""

_SNIPPET_MAX = 200
_CHUNK_CHARS = 400
_TOTAL_CONTEXT_CAP = 1_500


def _format_context(chunks: List[Document]) -> str:
    parts, total = [], 0
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        content = chunk.page_content[:_CHUNK_CHARS]
        entry = (
            f"[{i+1}] (Source: {meta.get('source_file','?')}, "
            f"Page: {int(meta.get('page',0))+1})\n{content}"
        )
        if total + len(entry) > _TOTAL_CONTEXT_CAP:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _find_cited_sentence(chunk_text: str, answer_text: str, chunk_id: int) -> str:
    """Find the specific sentence in a chunk that supports the answer claim."""
    claim_pattern = re.compile(
        r"([^.!?]*(?:\[[^\]]+\])*[^.!?]*\[" + str(chunk_id) + r"\][^.!?]*[.!?]?)",
        re.IGNORECASE,
    )
    claim_matches = claim_pattern.findall(answer_text)

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "of", "in", "on", "at",
        "to", "for", "and", "or", "but", "with", "by", "from", "that",
    }

    if not claim_matches:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
        return sentences[0][:_SNIPPET_MAX] if sentences else chunk_text[:_SNIPPET_MAX]

    claim_text = re.sub(r"\[\d+\]", "", " ".join(claim_matches))
    claim_words = {
        w.lower().strip(".,;:")
        for w in claim_text.split()
        if len(w) > 3 and w.lower() not in stopwords
    }

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if len(s.strip()) > 15]
    if not sentences:
        return chunk_text[:_SNIPPET_MAX]

    best, best_score = sentences[0], 0
    for sent in sentences:
        sent_words = {w.lower().strip(".,;:") for w in sent.split() if len(w) > 3}
        score = len(claim_words & sent_words)
        if score > best_score:
            best_score, best = score, sent

    return best[: _SNIPPET_MAX + 100]


def _extract_citations(chunks: List[Document], answer_text: str) -> List[Citation]:
    """Map [N] markers in the answer to sentence-level evidence from chunks."""
    referenced = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}
    all_citations = []

    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        chunk_text = chunk.page_content.replace("\n", " ").strip()
        snippet = _find_cited_sentence(chunk_text, answer_text, i + 1)

        all_citations.append(
            Citation(
                chunk_id=i + 1,
                page=int(meta.get("page", 0)) + 1,
                source_file=meta.get("source_file", "unknown"),
                snippet=snippet,
                full_chunk=chunk_text[: _SNIPPET_MAX * 3],
                similarity_score=meta.get("rerank_score"),
            )
        )

    if referenced:
        return [c for c in all_citations if c.chunk_id in referenced]
    return all_citations  # graceful fallback when LLM omits [N] markers


def generate_answer(
    query: str,
    chunks: List[Document],
    tier: str,
) -> Tuple[str, List[Citation]]:
    """Generate grounded answer and extract sentence-level citations."""
    llm = get_llm(tier=tier)
    context = _format_context(chunks)
    prompt = _ANSWER_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt)
    answer = response.content.strip()
    return answer, _extract_citations(chunks, answer)


def stream_answer(
    query: str,
    chunks: List[Document],
    tier: str,
) -> Generator[str, None, None]:
    """Stream answer tokens for SSE endpoint."""
    llm = get_llm(tier=tier)
    context = _format_context(chunks)
    prompt = _ANSWER_PROMPT.format(query=query, context=context)
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


def generate_node(state: RAGState) -> dict:
    """Generate the final grounded answer."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    tier = state.get("tier", "fast")

    answer, citations = generate_answer(query, chunks, tier)
    citation_dicts = [c.model_dump() for c in citations]

    logger.info("Generate: tier=%s citations=%d", tier, len(citations))

    return {
        "answer": answer,
        "citations": citation_dicts,
        "reasoning_trace": [
            f"[Generate] tier={tier} · {len(citations)} citations"
        ],
    }
