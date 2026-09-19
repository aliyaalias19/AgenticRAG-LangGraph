"""Stage 1: generate evaluation questions from sampled documents."""

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.eval.models import GeneratedQuestion, GenerationRun, QuestionType
from agentic_rag.eval.prompts import (
    GENERATION_SYSTEM,
    PROMPT_VERSION,
    build_generation_prompt,
)
from agentic_rag.ingest.models import Document
from agentic_rag.llm.client import LLMClient
from agentic_rag.llm.parsing import ParseError, parse_string_list
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

TYPE_CYCLE: tuple[QuestionType, ...] = (
    "direct",
    "direct",
    "direct",
    "paraphrased",
    "paraphrased",
    "paraphrased",
    "multi_hop",
    "multi_hop",
    "scenario",
    "scenario",
)


def assign_question_type(index: int) -> QuestionType:
    """Return the question type for the document at ``index``.

    Cycles through a fixed sequence giving a 30/30/20/20 split.
    """
    return TYPE_CYCLE[index % len(TYPE_CYCLE)]


def generate_for_document(
    document: Document,
    question_type: QuestionType,
    count: int,
    client: LLMClient,
    max_chars: int,
) -> list[GeneratedQuestion]:
    """Generate questions of one type for a single document."""
    prompt = build_generation_prompt(
        question_type=question_type,
        count=count,
        title=document.title,
        section=document.section_path[0] if document.section_path else "other",
        content=document.content[:max_chars],
    )

    response = client.complete(prompt, system=GENERATION_SYSTEM)

    try:
        texts = parse_string_list(response, key="questions")
    except ParseError as error:
        logger.warning(
            "question_parse_failed",
            doc_id=document.doc_id,
            question_type=question_type,
            error=str(error),
        )
        return []

    return [
        GeneratedQuestion(
            question_id=GeneratedQuestion.make_id(text, document.doc_id),
            question=text,
            question_type=question_type,
            source_doc_id=document.doc_id,
            source_section=(document.section_path[0] if document.section_path else "other"),
            language=document.language,
        )
        for text in texts[:count]
    ]


def generate_questions(
    documents: list[Document],
    client: LLMClient,
    settings: Settings | None = None,
) -> tuple[list[GeneratedQuestion], GenerationRun]:
    """Generate questions across a sampled document set."""
    settings = settings or get_settings()

    questions: list[GeneratedQuestion] = []
    failures = 0

    for index, document in enumerate(documents):
        question_type = assign_question_type(index)
        generated = generate_for_document(
            document=document,
            question_type=question_type,
            count=settings.eval.questions_per_document,
            client=client,
            max_chars=settings.eval.max_document_chars,
        )

        if not generated:
            failures += 1
        questions.extend(generated)

        if (index + 1) % 10 == 0:
            logger.info(
                "generation_progress",
                completed=index + 1,
                total=len(documents),
                questions=len(questions),
            )

    seen: set[str] = set()
    unique: list[GeneratedQuestion] = []
    for question in questions:
        if question.question_id not in seen:
            seen.add(question.question_id)
            unique.append(question)

    run = GenerationRun(
        model=settings.llm.model,
        prompt_version=PROMPT_VERSION,
        documents_sampled=len(documents),
        questions_generated=len(unique),
        input_tokens=client.usage.input_tokens,
        output_tokens=client.usage.output_tokens,
        random_seed=settings.random_seed,
    )

    logger.info(
        "generation_completed",
        questions=len(unique),
        duplicates_dropped=len(questions) - len(unique),
        documents_failed=failures,
        input_tokens=client.usage.input_tokens,
        output_tokens=client.usage.output_tokens,
    )
    return unique, run
