"""Stage 3: independently verify gold chunk labels."""

from agentic_rag.eval.models import LabelledQuestion, VerifiedQuestion
from agentic_rag.eval.overlap import overlap_ratio
from agentic_rag.eval.prompts import VERIFICATION_SYSTEM, build_verification_prompt
from agentic_rag.ingest.models import Chunk
from agentic_rag.llm.client import LLMClient
from agentic_rag.llm.parsing import ParseError, parse_json_object
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


def verify_chunk(question: str, chunk: Chunk, client: LLMClient) -> bool:
    """Return True if the chunk supports an answer to the question."""
    prompt = build_verification_prompt(question, chunk.content)

    try:
        response = client.complete(prompt, system=VERIFICATION_SYSTEM)
        parsed = parse_json_object(response)
    except ParseError as error:
        logger.warning("verification_parse_failed", chunk_id=chunk.chunk_id, error=str(error))
        return True

    return bool(parsed.get("supports", False))


def verify_question(
    question: LabelledQuestion,
    chunks_by_id: dict[str, Chunk],
    client: LLMClient,
) -> VerifiedQuestion:
    """Verify each gold chunk and compute the question's lexical overlap."""
    kept: list[str] = []
    rejected: list[str] = []

    for chunk_id in question.gold_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            rejected.append(chunk_id)
            logger.warning("verification_chunk_missing", chunk_id=chunk_id)
            continue

        if verify_chunk(question.question, chunk, client):
            kept.append(chunk_id)
        else:
            rejected.append(chunk_id)

    passages = [chunks_by_id[cid].content for cid in kept if cid in chunks_by_id]

    return VerifiedQuestion(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        source_doc_id=question.source_doc_id,
        source_section=question.source_section,
        language=question.language,
        gold_chunk_ids=kept,
        rejected_chunk_ids=rejected,
        lexical_overlap=round(overlap_ratio(question.question, passages), 4),
    )


def verify_questions(
    questions: list[LabelledQuestion],
    chunks: list[Chunk],
    client: LLMClient,
) -> list[VerifiedQuestion]:
    """Verify every labelled question's gold chunks."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    verified: list[VerifiedQuestion] = []
    for index, question in enumerate(questions):
        verified.append(verify_question(question, chunks_by_id, client))

        if (index + 1) % 25 == 0:
            logger.info("verification_progress", completed=index + 1, total=len(questions))

    answerable = [q for q in verified if q.is_answerable]
    rejected_total = sum(len(q.rejected_chunk_ids) for q in verified)
    kept_total = sum(len(q.gold_chunk_ids) for q in verified)

    logger.info(
        "verification_completed",
        questions=len(verified),
        answerable=len(answerable),
        chunks_kept=kept_total,
        chunks_rejected=rejected_total,
        mean_overlap=(
            round(sum(q.lexical_overlap for q in answerable) / len(answerable), 3)
            if answerable
            else 0.0
        ),
    )
    return verified
