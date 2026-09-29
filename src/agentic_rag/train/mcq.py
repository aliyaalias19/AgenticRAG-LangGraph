"""Multiple-choice evaluation for domain task accuracy.

Task accuracy is measured with four-option multiple choice rather than an
LLM judge. The reason is defensibility: MCQ scoring is deterministic, costs
nothing to re-run, and has no judge variance, so a reported improvement
reflects the model rather than the grader. The cost is that MCQ measures
discrimination rather than generation quality, which is why it sits alongside
the RAGAS metrics rather than replacing them.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag.llm.provider import ChatProvider, Message
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

MCQ_FILENAME = "mcq_heldout.json"
OPTION_LABELS = ("A", "B", "C", "D")

MCQ_SYSTEM = (
    "Answer the multiple-choice question using only the passages provided. "
    "Reply with a single letter: A, B, C or D. No explanation."
)

ANSWER_PATTERN = re.compile(r"\b([ABCD])\b")


@dataclass
class MCQItem:
    """One four-option question with its grounding passages."""

    item_id: str
    question: str
    options: list[str]
    correct_index: int
    passages: list[str] = field(default_factory=list)
    source_doc_id: str = ""

    @property
    def correct_label(self) -> str:
        return OPTION_LABELS[self.correct_index]

    def to_prompt(self) -> str:
        """Return the user turn for this item."""
        choices = "\n".join(
            f"{OPTION_LABELS[i]}. {option}" for i, option in enumerate(self.options)
        )
        passages = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(self.passages, start=1))
        return f"Passages:\n{passages}\n\nQuestion: {self.question}\n\n{choices}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_choice(response: str) -> str | None:
    """Extract the chosen option letter from a model response.

    Models sometimes answer "B." or "The answer is C". Taking the first
    standalone letter recovers those without accepting a letter that merely
    appears inside a word.
    """
    match = ANSWER_PATTERN.search(response.strip().upper())
    return match.group(1) if match else None


@dataclass
class MCQResult:
    """Score for one MCQ evaluation run."""

    label: str
    total: int
    correct: int
    unparseable: int
    per_item: dict[str, bool] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total": self.total,
            "correct": self.correct,
            "unparseable": self.unparseable,
            "accuracy": round(self.accuracy, 4),
        }


def evaluate_mcq(items: list[MCQItem], provider: ChatProvider, label: str) -> MCQResult:
    """Score a provider against the MCQ set."""
    correct = 0
    unparseable = 0
    per_item: dict[str, bool] = {}

    for index, item in enumerate(items):
        response = provider.complete(
            [Message("user", item.to_prompt())], system=MCQ_SYSTEM, max_tokens=8
        )
        choice = parse_choice(response.text)

        if choice is None:
            unparseable += 1
            per_item[item.item_id] = False
            continue

        is_correct = choice == item.correct_label
        per_item[item.item_id] = is_correct
        correct += int(is_correct)

        if (index + 1) % 25 == 0:
            logger.info("mcq_progress", label=label, done=index + 1, total=len(items))

    result = MCQResult(
        label=label,
        total=len(items),
        correct=correct,
        unparseable=unparseable,
        per_item=per_item,
    )
    logger.info("mcq_completed", **result.as_dict())
    return result


def compare_results(base: MCQResult, tuned: MCQResult) -> dict[str, Any]:
    """Return the accuracy delta and the per-item movement between two runs."""
    improved = [
        item_id
        for item_id, ok in tuned.per_item.items()
        if ok and not base.per_item.get(item_id, False)
    ]
    regressed = [
        item_id
        for item_id, ok in tuned.per_item.items()
        if not ok and base.per_item.get(item_id, False)
    ]
    return {
        "base": base.as_dict(),
        "tuned": tuned.as_dict(),
        "absolute_gain": round(tuned.accuracy - base.accuracy, 4),
        "relative_gain": (
            round((tuned.accuracy - base.accuracy) / base.accuracy, 4) if base.accuracy else 0.0
        ),
        "items_improved": len(improved),
        "items_regressed": len(regressed),
    }


def write_items(items: list[MCQItem], destination: Path) -> None:
    """Write the MCQ set as formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([i.as_dict() for i in items], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("mcq_set_written", path=str(destination), items=len(items))


def read_items(source: Path) -> list[MCQItem]:
    """Read an MCQ set from disk."""
    return [MCQItem(**payload) for payload in json.loads(source.read_text("utf-8"))]
