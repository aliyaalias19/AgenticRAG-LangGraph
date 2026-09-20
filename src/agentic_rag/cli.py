"""Command-line entry points for corpus operations."""

import argparse
import sys

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.generator import generate_questions
from agentic_rag.eval.labeller import label_questions
from agentic_rag.eval.models import LabelledSet, QuestionSet
from agentic_rag.eval.sampling import sample_documents
from agentic_rag.eval.storage import (
    LABELLED_FILENAME,
    QUESTIONS_FILENAME,
    read_question_set,
    write_labelled_set,
    write_question_set,
)
from agentic_rag.ingest.pipeline import (
    CHUNKS_FILENAME,
    CORPUS_FILENAME,
    ingest_corpus,
    read_chunks,
    read_corpus,
)
from agentic_rag.llm.client import LLMClient
from agentic_rag.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the requested CLI command and return a process exit code."""
    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Clone and process the document corpus")
    subparsers.add_parser("generate-questions", help="Generate evaluation questions")
    subparsers.add_parser("label-questions", help="Identify gold chunks for questions")

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "ingest":
        manifest = ingest_corpus()
        logger.info("ingest_command_finished", corpus_hash=manifest.corpus_hash)
        return 0

    if args.command == "generate-questions":
        settings = get_settings()
        settings.paths.ensure_exists()

        documents = read_corpus(settings.paths.processed_dir / CORPUS_FILENAME)
        sampled = sample_documents(
            documents,
            total=settings.eval.documents_to_sample,
            section_weights=settings.eval.section_weights,
            min_chars=settings.eval.min_document_chars,
            max_chars=settings.eval.max_document_chars,
            seed=settings.random_seed,
        )

        client = LLMClient(settings=settings)
        questions, run = generate_questions(sampled, client, settings)

        write_question_set(
            QuestionSet(run=run, questions=questions),
            settings.paths.evalsets_dir / QUESTIONS_FILENAME,
        )
        logger.info("generate_questions_finished", count=len(questions))
        return 0

    if args.command == "label-questions":
        settings = get_settings()

        question_set = read_question_set(settings.paths.evalsets_dir / QUESTIONS_FILENAME)
        chunks = read_chunks(settings.paths.processed_dir / CHUNKS_FILENAME)

        client = LLMClient(settings=settings)
        labelled = label_questions(
            question_set.questions,
            chunks,
            client,
            settings,
            checkpoint_path=settings.paths.evalsets_dir / "labelled_checkpoint.json",
        )

        write_labelled_set(
            LabelledSet(run=question_set.run, questions=labelled),
            settings.paths.evalsets_dir / LABELLED_FILENAME,
        )
        logger.info("label_questions_finished", count=len(labelled))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
