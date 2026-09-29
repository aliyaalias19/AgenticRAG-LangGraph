"""Instruction dataset construction for supervised fine-tuning.

Training examples are built from the same corpus and the same retrieval stack
the model will be served behind, so the model learns the format it will
actually be asked to produce: an answer grounded in numbered passages, with
inline citations, and an abstention when the passages do not support one.

The held-out evaluation set is split by *document*, not by example. Splitting
by example leaks: two questions generated from one document share phrasing and
facts, so a random split would put near-duplicates on both sides and inflate
the measured gain.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

TRAIN_FILENAME = "sft_train.jsonl"
HELDOUT_FILENAME = "sft_heldout.jsonl"

SYSTEM_PROMPT = (
    "You answer questions about Kubernetes using only the numbered passages "
    "provided. Cite passages inline as [1], [2]. If the passages do not "
    "contain the answer, say so instead of guessing."
)

ABSTENTION_RESPONSE = (
    "The provided passages do not contain enough information to answer this question."
)


@dataclass
class InstructionExample:
    """One supervised instruction/response pair."""

    example_id: str
    instruction: str
    response: str
    source_doc_id: str
    question_type: str
    is_abstention: bool = False

    def to_messages(self) -> list[dict[str, str]]:
        """Return the example in chat format for the SFT trainer."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.instruction},
            {"role": "assistant", "content": self.response},
        ]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = self.to_messages()
        return payload


def build_instruction(question: str, passages: list[str]) -> str:
    """Return the user turn for a training example."""
    numbered = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(passages, start=1))
    return f"Question: {question}\n\nPassages:\n{numbered}"


@dataclass
class DatasetSplit:
    """A document-disjoint train and held-out split."""

    train: list[InstructionExample] = field(default_factory=list)
    heldout: list[InstructionExample] = field(default_factory=list)

    @property
    def train_documents(self) -> set[str]:
        return {e.source_doc_id for e in self.train}

    @property
    def heldout_documents(self) -> set[str]:
        return {e.source_doc_id for e in self.heldout}

    @property
    def is_disjoint(self) -> bool:
        """Return True when no document appears on both sides."""
        return not (self.train_documents & self.heldout_documents)

    def summary(self) -> dict[str, Any]:
        return {
            "train_examples": len(self.train),
            "heldout_examples": len(self.heldout),
            "train_documents": len(self.train_documents),
            "heldout_documents": len(self.heldout_documents),
            "document_disjoint": self.is_disjoint,
            "abstentions_in_train": sum(1 for e in self.train if e.is_abstention),
        }


def split_by_document(
    examples: list[InstructionExample],
    heldout_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Split examples so that no document appears in both halves."""
    documents = sorted({e.source_doc_id for e in examples})
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    rng.shuffle(documents)

    cut = int(len(documents) * heldout_fraction)
    heldout_documents = set(documents[:cut])

    split = DatasetSplit(
        train=[e for e in examples if e.source_doc_id not in heldout_documents],
        heldout=[e for e in examples if e.source_doc_id in heldout_documents],
    )
    logger.info("dataset_split", **split.summary())
    return split


def write_jsonl(examples: list[InstructionExample], destination: Path) -> None:
    """Write examples as JSON Lines."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
    logger.info("dataset_written", path=str(destination), examples=len(examples))


def read_jsonl(source: Path) -> list[InstructionExample]:
    """Read examples from JSON Lines."""
    examples = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload.pop("messages", None)
        examples.append(InstructionExample(**payload))
    return examples
