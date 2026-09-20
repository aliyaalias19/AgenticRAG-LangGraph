"""Typed models for evaluation questions and sets."""

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["direct", "paraphrased", "multi_hop", "scenario"]


class GeneratedQuestion(BaseModel):
    """A question produced by stage 1, before labelling or verification."""

    question_id: str
    question: str
    question_type: QuestionType
    source_doc_id: str
    source_section: str
    language: str = Field(default="en")

    @staticmethod
    def make_id(question: str, doc_id: str) -> str:
        """Return a stable identifier derived from the question and its source."""
        digest = hashlib.sha256(f"{doc_id}|{question}".encode()).hexdigest()
        return f"q_{digest[:16]}"


class GenerationRun(BaseModel):
    """Provenance for one question-generation run."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    prompt_version: str
    documents_sampled: int = Field(ge=0)
    questions_generated: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    random_seed: int


class QuestionSet(BaseModel):
    """A generated question set with its provenance."""

    run: GenerationRun
    questions: list[GeneratedQuestion] = Field(default_factory=list)


class LabelledQuestion(BaseModel):
    """A question with its gold chunks identified."""

    question_id: str
    question: str
    question_type: QuestionType
    source_doc_id: str
    source_section: str
    language: str = Field(default="en")
    gold_chunk_ids: list[str] = Field(default_factory=list)
    labeller_reasoning: str = Field(default="")

    @property
    def is_answerable(self) -> bool:
        """Return True when at least one supporting chunk was identified."""
        return bool(self.gold_chunk_ids)


class LabelledSet(BaseModel):
    """A labelled question set with its provenance."""

    run: GenerationRun
    questions: list[LabelledQuestion] = Field(default_factory=list)
