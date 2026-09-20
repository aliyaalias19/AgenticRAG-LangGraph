"""Persistence for evaluation question sets."""

import json
from pathlib import Path

from agentic_rag.eval.models import LabelledSet, QuestionSet
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

QUESTIONS_FILENAME = "questions_raw.json"
LABELLED_FILENAME = "questions_labelled.json"


def write_question_set(question_set: QuestionSet, destination: Path) -> None:
    """Write a question set as formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(question_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "question_set_written",
        path=str(destination),
        count=len(question_set.questions),
    )


def read_question_set(source: Path) -> QuestionSet:
    """Read a question set from disk."""
    return QuestionSet.model_validate_json(source.read_text(encoding="utf-8"))


def write_labelled_set(labelled_set: LabelledSet, destination: Path) -> None:
    """Write a labelled question set as formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(labelled_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "labelled_set_written",
        path=str(destination),
        count=len(labelled_set.questions),
    )


def read_labelled_set(source: Path) -> LabelledSet:
    """Read a labelled question set from disk."""
    return LabelledSet.model_validate_json(source.read_text(encoding="utf-8"))
