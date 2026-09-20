"""Stage 2: identify gold chunks for generated questions."""

import json
from collections import defaultdict
from pathlib import Path

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.eval.models import GeneratedQuestion, LabelledQuestion
from agentic_rag.eval.prompts import LABELLING_SYSTEM, build_labelling_prompt
from agentic_rag.ingest.models import Chunk
from agentic_rag.llm.client import LLMClient
from agentic_rag.llm.parsing import ParseError, parse_json_object
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

MAX_GOLD_CHUNKS = 3


def index_chunks_by_document(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their parent document, preserving chunk order."""
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    for doc_chunks in grouped.values():
        doc_chunks.sort(key=lambda c: c.chunk_index)
    return dict(grouped)


def _parse_passage_numbers(response: str, passage_count: int) -> tuple[list[int], str]:
    """Extract valid 1-based passage numbers and the labeller's reasoning."""
    parsed = parse_json_object(response)

    raw = parsed.get("passage_numbers")
    if not isinstance(raw, list):
        message = "Response is missing a passage_numbers list"
        raise ParseError(message)

    numbers = [value for value in raw if isinstance(value, int) and 1 <= value <= passage_count]
    reasoning = parsed.get("reasoning", "")
    return numbers, reasoning if isinstance(reasoning, str) else ""


def label_question(
    question: GeneratedQuestion,
    chunks: list[Chunk],
    client: LLMClient,
) -> LabelledQuestion:
    """Identify which of a document's chunks answer the given question."""
    passages = [chunk.content for chunk in chunks]
    prompt = build_labelling_prompt(question.question, passages)

    try:
        response = client.complete(prompt, system=LABELLING_SYSTEM)
        numbers, reasoning = _parse_passage_numbers(response, len(passages))
    except ParseError as error:
        logger.warning(
            "labelling_parse_failed",
            question_id=question.question_id,
            error=str(error),
        )
        numbers, reasoning = [], ""

    gold_ids = [chunks[number - 1].chunk_id for number in numbers[:MAX_GOLD_CHUNKS]]

    return LabelledQuestion(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        source_doc_id=question.source_doc_id,
        source_section=question.source_section,
        language=question.language,
        gold_chunk_ids=gold_ids,
        labeller_reasoning=reasoning,
    )


def label_questions(
    questions: list[GeneratedQuestion],
    chunks: list[Chunk],
    client: LLMClient,
    settings: Settings | None = None,
    checkpoint_path: Path | None = None,
) -> list[LabelledQuestion]:
    """Label every question with the chunks that answer it."""
    settings = settings or get_settings()
    by_document = index_chunks_by_document(chunks)

    labelled: list[LabelledQuestion] = []
    missing_documents = 0

    for index, question in enumerate(questions):
        doc_chunks = by_document.get(question.source_doc_id)
        if not doc_chunks:
            missing_documents += 1
            logger.warning(
                "labelling_document_missing",
                question_id=question.question_id,
                doc_id=question.source_doc_id,
            )
            continue

        labelled.append(label_question(question, doc_chunks, client))

        if checkpoint_path is not None and (index + 1) % 25 == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    [q.model_dump(mode="json") for q in labelled],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        if (index + 1) % 25 == 0:
            logger.info("labelling_progress", completed=index + 1, total=len(questions))

    answerable = sum(1 for q in labelled if q.is_answerable)
    gold_counts = [len(q.gold_chunk_ids) for q in labelled if q.is_answerable]

    logger.info(
        "labelling_completed",
        labelled=len(labelled),
        answerable=answerable,
        unanswerable=len(labelled) - answerable,
        missing_documents=missing_documents,
        mean_gold_chunks=(round(sum(gold_counts) / len(gold_counts), 2) if gold_counts else 0),
    )
    return labelled
