"""Offline retrieval evaluation across configurations.

Every configuration is scored on the same frozen question set, so the only
variable between runs is the retrieval policy. This is a controlled experiment
rather than an end-to-end LLM judgement: an LLM scoring whole answers cannot
tell you whether a regression came from retrieval depth, fusion weights or the
reranker cutoff, and those are exactly the knobs being tuned.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from agentic_rag.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from agentic_rag.obs.logging import get_logger
from agentic_rag.retrieval.hybrid import RetrievalConfig

logger = get_logger(__name__)


class EvalQuestion(Protocol):
    """Minimal interface the harness needs from an evaluation question."""

    question_id: str
    question: str
    question_type: str
    gold_chunk_ids: list[str]

    @property
    def is_answerable(self) -> bool: ...


@dataclass
class EvalResult:
    """Aggregate metrics for one retrieval configuration."""

    config_name: str
    signals: str
    questions: int
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    hit_at_10: float
    mrr: float
    ndcg_at_10: float
    by_question_type: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float):
                payload[key] = round(value, 4)
        payload["by_question_type"] = {k: round(v, 4) for k, v in self.by_question_type.items()}
        return payload

    def summary_line(self) -> str:
        """Return a single fixed-width row for console output."""
        return (
            f"{self.config_name:<26} {self.recall_at_5:>7.3f} "
            f"{self.recall_at_10:>7.3f} {self.precision_at_5:>7.3f} "
            f"{self.mrr:>7.3f} {self.ndcg_at_10:>7.3f}"
        )


HEADER = f"{'config':<26} {'R@5':>7} {'R@10':>7} {'P@5':>7} {'MRR':>7} {'nDCG':>7}"


def evaluate_config(
    questions: Sequence[Any],
    retriever: Any,
    config: RetrievalConfig,
    context: Any = None,
    language: str | None = None,
) -> EvalResult:
    """Run one configuration over the evaluation set."""
    answerable = [q for q in questions if q.is_answerable]

    buckets: dict[str, list[float]] = {"r5": [], "r10": [], "p5": [], "h10": [], "rr": [], "nd": []}
    per_type: dict[str, list[float]] = {}

    for index, question in enumerate(answerable):
        hits = retriever.retrieve(question.question, config, context=context, language=language)
        retrieved = [h.chunk_id for h in hits]
        gold = list(question.gold_chunk_ids)

        recall10 = recall_at_k(retrieved, gold, 10)
        buckets["r5"].append(recall_at_k(retrieved, gold, 5))
        buckets["r10"].append(recall10)
        buckets["p5"].append(precision_at_k(retrieved, gold, 5))
        buckets["h10"].append(hit_at_k(retrieved, gold, 10))
        buckets["rr"].append(reciprocal_rank(retrieved, gold))
        buckets["nd"].append(ndcg_at_k(retrieved, gold, 10))
        per_type.setdefault(str(question.question_type), []).append(recall10)

        if (index + 1) % 50 == 0:
            logger.info(
                "eval_progress",
                config=config.name,
                done=index + 1,
                total=len(answerable),
            )

    result = EvalResult(
        config_name=config.name,
        signals=",".join(config.signals),
        questions=len(answerable),
        recall_at_5=mean(buckets["r5"]) if buckets["r5"] else 0.0,
        recall_at_10=mean(buckets["r10"]) if buckets["r10"] else 0.0,
        precision_at_5=mean(buckets["p5"]) if buckets["p5"] else 0.0,
        hit_at_10=mean(buckets["h10"]) if buckets["h10"] else 0.0,
        mrr=mean(buckets["rr"]) if buckets["rr"] else 0.0,
        ndcg_at_10=mean(buckets["nd"]) if buckets["nd"] else 0.0,
        by_question_type={k: mean(v) for k, v in sorted(per_type.items())},
    )
    logger.info("eval_completed", **result.as_dict())
    return result


def write_results(results: list[EvalResult], destination: Path) -> None:
    """Write evaluation results as formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n", encoding="utf-8"
    )
    logger.info("eval_results_written", path=str(destination), configs=len(results))


def print_table(results: list[EvalResult]) -> None:
    """Print a comparison table and the improvement over the first row."""
    print(f"\n{HEADER}")
    print("-" * len(HEADER))
    for result in results:
        print(result.summary_line())

    if len(results) >= 2:
        baseline, best = results[0], max(results, key=lambda r: r.recall_at_10)
        delta = best.recall_at_10 - baseline.recall_at_10
        print(
            f"\nRecall@10  {baseline.config_name} {baseline.recall_at_10:.3f}"
            f"  ->  {best.config_name} {best.recall_at_10:.3f}"
            f"  ({delta:+.3f})"
        )

    print("\nrecall@10 by question type:")
    for result in results:
        types = "  ".join(f"{k}={v:.3f}" for k, v in result.by_question_type.items())
        print(f"  {result.config_name:<26} {types}")
