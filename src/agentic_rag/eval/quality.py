"""Answer quality metrics in the RAGAS family.

Four metrics, each isolating one failure mode:

- context precision: did retrieval put the useful passages near the top
- context recall:    did retrieval find everything the answer needed
- faithfulness:      is every claim in the answer supported by the context
- answer relevance:  does the answer address the question that was asked

They are deliberately separable. An answer can be perfectly faithful to
irrelevant context, or perfectly relevant while inventing its facts, and a
single blended score hides which of those is happening.

Context precision and recall are computed arithmetically from gold labels
where those exist; faithfulness and answer relevance need a judge, since no
label can anticipate the wording of a generated answer.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.eval.metrics import average_precision, recall_at_k
from agentic_rag.llm.parsing import ParseError, clamp, parse_json_object
from agentic_rag.llm.provider import ChatProvider, Message
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

FAITHFULNESS_SYSTEM = """\
You decompose an answer into atomic factual claims and check each against the \
provided context.

A claim is supported when the context states it or directly entails it. \
Ignore hedges, restatements of the question, and statements that the context \
does not contain something.

Return only valid JSON. No preamble, no markdown fences.\
"""

FAITHFULNESS_TEMPLATE = """\
Context:
{context}

Answer:
{answer}

Return JSON in exactly this shape:
{{"claims": [{{"claim": "...", "supported": true}}]}}\
"""

RELEVANCE_SYSTEM = """\
You judge whether an answer addresses the question that was asked.

Score 0.0 to 1.0:
- 1.0  answers the question directly and completely
- 0.6  answers part of it, or answers it indirectly
- 0.3  discusses the topic without answering
- 0.0  does not address the question

An honest refusal to answer, when the context genuinely lacks the \
information, scores 0.5: it is the correct behaviour, but it is not an answer.

Return only valid JSON. No preamble, no markdown fences.\
"""

RELEVANCE_TEMPLATE = """\
Question: {question}

Answer:
{answer}

Return JSON in exactly this shape:
{{"relevance": 0.9, "reason": "one short sentence"}}\
"""


@dataclass(frozen=True)
class QualityScores:
    """The four quality signals for one answered question."""

    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
        }

    @property
    def harmonic_mean(self) -> float:
        """Return the harmonic mean, which one weak metric drags down.

        An arithmetic mean lets a strong retrieval score paper over a
        hallucinating generator. The harmonic mean does not.
        """
        values = [
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevance,
        ]
        if any(v <= 0.0 for v in values):
            return 0.0
        return len(values) / sum(1.0 / v for v in values)


@dataclass(frozen=True)
class ShippingThreshold:
    """Minimum acceptable score for each quality metric.

    A configuration ships only if it clears every threshold. Requiring all
    four prevents the usual failure, where one metric is tuned upward while
    another quietly collapses.
    """

    context_precision: float = 0.60
    context_recall: float = 0.75
    faithfulness: float = 0.85
    answer_relevance: float = 0.80

    def failures(self, scores: QualityScores) -> dict[str, tuple[float, float]]:
        """Return {metric: (observed, required)} for every metric below target."""
        checks = {
            "context_precision": (scores.context_precision, self.context_precision),
            "context_recall": (scores.context_recall, self.context_recall),
            "faithfulness": (scores.faithfulness, self.faithfulness),
            "answer_relevance": (scores.answer_relevance, self.answer_relevance),
        }
        return {
            name: (observed, required)
            for name, (observed, required) in checks.items()
            if observed < required
        }

    def passes(self, scores: QualityScores) -> bool:
        """Return True when every metric clears its threshold."""
        return not self.failures(scores)


def context_precision(retrieved: list[str], gold: list[str]) -> float:
    """Rank-weighted precision of the retrieved context.

    Mean average precision is used rather than flat precision because a
    reader, and a context window, both care where the useful passage sits.
    """
    return average_precision(retrieved, gold)


def context_recall(retrieved: list[str], gold: list[str]) -> float:
    """Fraction of gold passages present in the retrieved context."""
    return recall_at_k(retrieved, gold, len(retrieved) or 1)


def faithfulness(
    answer: str, context: list[str], provider: ChatProvider
) -> tuple[float, list[str]]:
    """Return the supported-claim fraction and the unsupported claims."""
    if not answer.strip() or not context:
        return 0.0, []

    response = provider.complete(
        [
            Message(
                "user",
                FAITHFULNESS_TEMPLATE.format(context="\n\n".join(context), answer=answer),
            )
        ],
        system=FAITHFULNESS_SYSTEM,
    )

    try:
        parsed = parse_json_object(response.text)
    except ParseError as error:
        logger.warning("faithfulness_parse_failed", error=str(error))
        return 0.0, []

    claims = [c for c in parsed.get("claims", []) if isinstance(c, dict)]
    if not claims:
        return 0.0, []

    unsupported = [str(c.get("claim", "")) for c in claims if not bool(c.get("supported", False))]
    return (len(claims) - len(unsupported)) / len(claims), unsupported


def answer_relevance(question: str, answer: str, provider: ChatProvider) -> float:
    """Return how well the answer addresses the question."""
    if not answer.strip():
        return 0.0

    response = provider.complete(
        [Message("user", RELEVANCE_TEMPLATE.format(question=question, answer=answer))],
        system=RELEVANCE_SYSTEM,
    )
    try:
        return clamp(parse_json_object(response.text).get("relevance"))
    except ParseError as error:
        logger.warning("relevance_parse_failed", error=str(error))
        return 0.0


def score_answer(
    question: str,
    answer: str,
    retrieved: list[str],
    gold: list[str],
    context: list[str],
    provider: ChatProvider,
) -> QualityScores:
    """Compute all four quality metrics for one answered question."""
    supported, _ = faithfulness(answer, context, provider)
    return QualityScores(
        context_precision=context_precision(retrieved, gold),
        context_recall=context_recall(retrieved, gold),
        faithfulness=supported,
        answer_relevance=answer_relevance(question, answer, provider),
    )


@dataclass
class QualityAggregate:
    """Mean quality scores across an evaluation run."""

    scores: list[QualityScores] = field(default_factory=list)

    def add(self, score: QualityScores) -> None:
        self.scores.append(score)

    def mean(self) -> QualityScores:
        """Return the element-wise mean of every recorded score."""
        if not self.scores:
            return QualityScores()
        return QualityScores(
            context_precision=statistics.mean(s.context_precision for s in self.scores),
            context_recall=statistics.mean(s.context_recall for s in self.scores),
            faithfulness=statistics.mean(s.faithfulness for s in self.scores),
            answer_relevance=statistics.mean(s.answer_relevance for s in self.scores),
        )

    def report(self, threshold: ShippingThreshold | None = None) -> dict[str, Any]:
        """Return the aggregate scores and their pass/fail verdict."""
        threshold = threshold or ShippingThreshold()
        mean = self.mean()
        failures = threshold.failures(mean)
        return {
            "questions": len(self.scores),
            **mean.as_dict(),
            "harmonic_mean": round(mean.harmonic_mean, 4),
            "ships": not failures,
            "failures": {
                name: {"observed": round(obs, 4), "required": req}
                for name, (obs, req) in failures.items()
            },
        }
