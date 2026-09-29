"""Score the base and fine-tuned models on the held-out MCQ set.

Both models are scored on the same items with the same prompt. The number
that matters is measured on the model that will actually be served -- the
merged, AWQ-quantised artefact -- not on the training checkpoint, because
merging and quantisation each move accuracy.

    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
    python scripts/run_mcq_eval.py --label base --model meta-llama/Llama-3.1-8B-Instruct

    vllm serve data/models/awq --quantization awq --port 8000
    python scripts/run_mcq_eval.py --label tuned-awq --model data/models/awq --compare
"""

import argparse
import json
import sys

from agentic_rag.config.settings import get_settings
from agentic_rag.llm.provider import OpenAICompatibleProvider
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.obs.tracking import track_run
from agentic_rag.train.mcq import (
    MCQ_FILENAME,
    MCQResult,
    compare_results,
    evaluate_mcq,
    read_items,
)

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    items = read_items(settings.paths.evalsets_dir / MCQ_FILENAME)
    if args.limit:
        items = items[: args.limit]

    provider = OpenAICompatibleProvider(
        settings, base_url=args.base_url, model=args.model, api_key=""
    )

    with track_run(f"mcq-{args.label}", settings, tags={"stage": "task"}) as tracker:
        tracker.log_params(label=args.label, model=args.model, items=len(items))
        result = evaluate_mcq(items, provider, args.label)
        tracker.log_metrics(
            accuracy=result.accuracy,
            correct=result.correct,
            unparseable=result.unparseable,
        )

    output = settings.paths.results_dir / f"mcq_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**result.as_dict(), "per_item": result.per_item}, indent=2),
        encoding="utf-8",
    )
    print(
        f"\n{args.label}: {result.accuracy:.1%} "
        f"({result.correct}/{result.total}, {result.unparseable} unparseable)"
    )

    if args.compare:
        base_path = settings.paths.results_dir / "mcq_base.json"
        if base_path.is_file():
            payload = json.loads(base_path.read_text(encoding="utf-8"))
            base = MCQResult(
                label=payload["label"],
                total=payload["total"],
                correct=payload["correct"],
                unparseable=payload["unparseable"],
                per_item=payload.get("per_item", {}),
            )
            comparison = compare_results(base, result)
            print(json.dumps(comparison, indent=2))
            (settings.paths.results_dir / "mcq_comparison.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )
        else:
            print("no base result found; run with --label base first")

    return 0


if __name__ == "__main__":
    sys.exit(main())
