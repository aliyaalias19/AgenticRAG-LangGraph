"""Score answer quality on the four RAGAS metrics and apply the ship gate.

Retrieval metrics say whether the evidence was found. These say whether the
answer built on it is any good. A configuration ships only if all four clear
their thresholds, which is what stops one metric being tuned at another's
expense.

    python scripts/run_quality_eval.py --limit 50
"""

import argparse
import json
import sys

from agentic_rag.agent.graph import AgentRunner
from agentic_rag.agent.nodes import GraphDeps
from agentic_rag.config.settings import get_settings
from agentic_rag.eval.quality import QualityAggregate, ShippingThreshold, score_answer
from agentic_rag.eval.storage import VERIFIED_FILENAME, read_verified_set
from agentic_rag.llm.provider import build_provider
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.obs.tracking import track_run
from agentic_rag.retrieval.hybrid import RetrievalConfig, build_retriever
from agentic_rag.security.tenancy import anonymous_context

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default="hybrid_rrf_rerank")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    questions = [
        q
        for q in read_verified_set(settings.paths.evalsets_dir / VERIFIED_FILENAME).questions
        if q.is_answerable
    ]
    if args.limit:
        questions = questions[: args.limit]

    provider = build_provider(settings)
    deps = GraphDeps(
        provider=provider,
        retriever=build_retriever(settings),
        settings=settings,
        retrieval_config=RetrievalConfig(args.label, use_bm25=True, use_reranker=True),
    )
    runner = AgentRunner(deps)
    context = anonymous_context(settings)
    aggregate = QualityAggregate()

    with track_run(f"quality-{args.label}", settings, tags={"stage": "quality"}) as tracker:
        tracker.log_params(config=args.label, questions=len(questions))

        for index, question in enumerate(questions):
            state = runner.run(question.question, context, question.question_id)
            retrieved = [h.chunk_id for h in state.get("context", [])]
            aggregate.add(
                score_answer(
                    question=question.question,
                    answer=str(state.get("answer", "")),
                    retrieved=retrieved,
                    gold=list(question.gold_chunk_ids),
                    context=[h.content for h in state.get("context", [])],
                    provider=provider,
                )
            )
            if (index + 1) % 10 == 0:
                logger.info("quality_progress", done=index + 1, total=len(questions))

        report = aggregate.report(ShippingThreshold())
        tracker.log_metrics(**{k: v for k, v in report.items() if isinstance(v, (int, float))})

    print(json.dumps(report, indent=2))
    output = settings.paths.results_dir / f"quality_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["ships"]:
        print("\nSHIP GATE FAILED")
        for metric, detail in report["failures"].items():
            print(f"  {metric}: {detail['observed']} < {detail['required']}")
        return 1
    print("\nship gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
