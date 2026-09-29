"""Benchmark retrieval configurations against the verified evaluation set.

This is a controlled experiment. Each configuration sees the same frozen
questions, the same corpus hash and the same gold labels, so the only thing
that varies is the retrieval policy. That is what makes a difference in
Recall@10 attributable to the change rather than to noise or drift.

    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --configs dense_only hybrid_rrf_rerank
"""

import argparse
import sys

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.retrieval_eval import evaluate_config, print_table, write_results
from agentic_rag.eval.storage import VERIFIED_FILENAME, read_verified_set
from agentic_rag.ingest.pipeline import MANIFEST_FILENAME
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.obs.tracking import track_run
from agentic_rag.retrieval.hybrid import DEFAULT_CONFIGS, build_retriever
from agentic_rag.security.tenancy import anonymous_context

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--language", default=None, choices=["en", "zh"])
    parser.add_argument("--output", default="retrieval_eval.json")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    questions = read_verified_set(settings.paths.evalsets_dir / VERIFIED_FILENAME).questions
    retriever = build_retriever(settings)
    context = anonymous_context(settings)

    selected = [c for c in DEFAULT_CONFIGS if not args.configs or c.name in args.configs]
    if not selected:
        parser.error(f"no configs matched; available: {[c.name for c in DEFAULT_CONFIGS]}")

    results = []
    for config in selected:
        with track_run(
            f"retrieval-{config.name}", settings, tags={"stage": "retrieval"}
        ) as tracker:
            tracker.log_corpus_provenance(settings.paths.processed_dir / MANIFEST_FILENAME)
            tracker.log_params(
                config=config.name,
                signals=list(config.signals),
                dense_candidates=config.dense_candidates,
                sparse_candidates=config.sparse_candidates,
                bm25_candidates=config.bm25_candidates,
                rrf_k=config.rrf_k,
                reranker=config.use_reranker,
                top_k=config.top_k,
                language=args.language or "all",
            )
            result = evaluate_config(questions, retriever, config, context, args.language)
            tracker.log_metrics(
                **{k: v for k, v in result.as_dict().items() if isinstance(v, (int, float))}
            )
            results.append(result)

    print_table(results)
    write_results(results, settings.paths.results_dir / args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
