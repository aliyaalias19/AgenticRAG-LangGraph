"""Benchmark a vLLM endpoint across concurrency levels.

    vllm serve data/models/awq \\
        --quantization awq --max-model-len 4096 --port 8000

    python scripts/bench_vllm.py --concurrency 1 2 4 8 16

Reports TTFT and inter-token latency at percentiles, per-stream decode rate,
and aggregate throughput, so the latency/throughput trade-off is visible
rather than collapsed into one number.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agentic_rag.config.settings import get_settings
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.serving.benchmark import BenchmarkSuite, run_level

logger = get_logger(__name__)

# Roughly 500 tokens of realistic RAG input: a system instruction plus several
# retrieved passages. Benchmarking on a short prompt understates TTFT badly,
# because prefill is where the input length is paid for.
PROMPT_TEMPLATE = """\
You answer questions about Kubernetes using only the numbered passages below.
Cite passages inline as [1], [2].

Passages:
{passages}

Question: What happens to DaemonSet pods when a node is drained, and which \
flag changes that behaviour?"""

PASSAGE = (
    "[{n}] When you run kubectl drain, the node is first cordoned so that the "
    "scheduler stops placing new pods on it. The command then evicts the pods "
    "that are already running, respecting any PodDisruptionBudget that applies "
    "to them. Pods managed by a DaemonSet are not evicted by default, because "
    "the DaemonSet controller would immediately recreate them on the same node. "
    "Passing --ignore-daemonsets allows the drain to proceed while leaving those "
    "pods in place. Pods with local storage require --delete-emptydir-data, "
    "since evicting them discards data that cannot be recovered."
)


def build_prompt(passage_count: int) -> str:
    """Return a benchmark prompt with ``passage_count`` retrieved passages."""
    passages = "\n\n".join(PASSAGE.format(n=i) for i in range(1, passage_count + 1))
    return PROMPT_TEMPLATE.format(passages=passages)


async def run_sweep(
    base_url: str,
    model: str,
    levels: list[int],
    requests_per_level: int,
    max_output_tokens: int,
    passage_count: int,
) -> BenchmarkSuite:
    """Run the full concurrency sweep."""
    prompt = build_prompt(passage_count)
    logger.info(
        "benchmark_started",
        model=model,
        levels=levels,
        approx_input_tokens=len(prompt) // 4,
    )

    suite = BenchmarkSuite()
    for concurrency in levels:
        result = await run_level(
            base_url=base_url,
            model=model,
            prompt=prompt,
            concurrency=concurrency,
            requests=max(requests_per_level, concurrency),
            max_output_tokens=max_output_tokens,
            label=f"c{concurrency}",
        )
        suite.add(result)
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults come from settings so the benchmark and the serving layer cannot
    # disagree about the endpoint. The served model name is what vLLM registers,
    # which is not the same as the path the weights were loaded from.
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--passages", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/results/vllm_benchmark.json"))
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    # An unset flag means "use the configured endpoint", not a hardcoded path.
    # The served model name is what vLLM registers and is not the directory the
    # weights were loaded from; defaulting to the path produces a silent 404.
    base_url = args.base_url or settings.llm.vllm_base_url
    model = args.model or settings.llm.vllm_model
    suite = asyncio.run(
        run_sweep(
            base_url,
            model,
            args.concurrency,
            args.requests,
            args.max_output_tokens,
            args.passages,
        )
    )

    suite.print_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite.as_dict(), indent=2) + "\n", "utf-8")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
