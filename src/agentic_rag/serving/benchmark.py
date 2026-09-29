"""Load generation and latency measurement for an OpenAI-compatible endpoint.

The numbers that matter for a serving layer are not averages. A mean TTFT of
400ms is consistent with every request being fast and with one in twenty
taking four seconds, and only the second case generates complaints. Everything
here is reported at percentiles.

Two metrics are easy to conflate and measure different things:

- TTFT is how long the user waits before anything appears. It is dominated by
  prefill, so it scales with input length and with how many other requests are
  queued ahead.
- inter-token latency is how fast text appears once it starts. It is decode
  bound, so it scales with concurrency far more gently than TTFT does.

Throughput and latency trade against each other: raising concurrency raises
aggregate tokens per second while making each individual stream slower. The
useful output of a sweep is that curve, not a single headline number.
"""

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestSample:
    """Timing for one completed streaming request."""

    ttft_seconds: float
    total_seconds: float
    output_tokens: int
    input_tokens: int
    succeeded: bool = True
    error: str = ""

    @property
    def inter_token_seconds(self) -> float:
        """Mean gap between tokens after the first."""
        remaining = self.output_tokens - 1
        if remaining <= 0:
            return 0.0
        return (self.total_seconds - self.ttft_seconds) / remaining

    @property
    def output_tokens_per_second(self) -> float:
        """Decode rate for this stream alone."""
        decode = self.total_seconds - self.ttft_seconds
        return (self.output_tokens - 1) / decode if decode > 0 else 0.0


def percentile(values: list[float], fraction: float) -> float:
    """Return the value at ``fraction`` through a sorted copy of ``values``.

    Nearest-rank rather than interpolated: with a few hundred samples,
    interpolation invents a latency no request actually experienced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


@dataclass
class BenchmarkResult:
    """Aggregate result for one concurrency level."""

    label: str
    concurrency: int
    input_tokens: int
    max_output_tokens: int
    requests: int
    failures: int
    wall_seconds: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    itl_p50: float
    itl_p95: float
    output_tokens_per_second_per_stream: float
    aggregate_output_tokens_per_second: float
    requests_per_second: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, float):
                payload[key] = round(value, 4)
        return payload

    def summary_line(self) -> str:
        return (
            f"{self.concurrency:>4}  {self.ttft_p50 * 1000:>8.0f}  "
            f"{self.ttft_p95 * 1000:>8.0f}  {self.ttft_p99 * 1000:>8.0f}  "
            f"{self.itl_p50 * 1000:>7.1f}  "
            f"{self.output_tokens_per_second_per_stream:>8.1f}  "
            f"{self.aggregate_output_tokens_per_second:>9.1f}"
        )


HEADER = (
    f"{'conc':>4}  {'TTFT p50':>8}  {'TTFT p95':>8}  {'TTFT p99':>8}  "
    f"{'ITL p50':>7}  {'tok/s/st':>8}  {'agg tok/s':>9}"
)


def summarise(
    label: str,
    concurrency: int,
    input_tokens: int,
    max_output_tokens: int,
    samples: list[RequestSample],
    wall_seconds: float,
) -> BenchmarkResult:
    """Aggregate raw samples into a reportable result."""
    ok = [s for s in samples if s.succeeded]
    ttfts = [s.ttft_seconds for s in ok]
    itls = [s.inter_token_seconds for s in ok if s.output_tokens > 1]
    per_stream = [s.output_tokens_per_second for s in ok if s.output_tokens > 1]
    total_output = sum(s.output_tokens for s in ok)

    return BenchmarkResult(
        label=label,
        concurrency=concurrency,
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        requests=len(samples),
        failures=len(samples) - len(ok),
        wall_seconds=wall_seconds,
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p95=percentile(ttfts, 0.95),
        ttft_p99=percentile(ttfts, 0.99),
        itl_p50=percentile(itls, 0.50),
        itl_p95=percentile(itls, 0.95),
        output_tokens_per_second_per_stream=(statistics.mean(per_stream) if per_stream else 0.0),
        aggregate_output_tokens_per_second=(total_output / wall_seconds if wall_seconds else 0.0),
        requests_per_second=len(ok) / wall_seconds if wall_seconds else 0.0,
    )


async def stream_once(
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> RequestSample:
    """Issue one streaming completion and time it."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    started = time.perf_counter()
    first_token_at: float | None = None
    # Counted from deltas, and overridden by the server's own usage block when
    # one arrives. Delta counting is an approximation -- a chunk can carry
    # more than one token -- so the reported figure is preferred where the
    # server provides it.
    delta_count = 0
    reported_output_tokens = 0
    input_tokens = 0

    try:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break

                body = json.loads(data)
                if usage := body.get("usage"):
                    input_tokens = int(usage.get("prompt_tokens", input_tokens))
                    reported_output_tokens = int(
                        usage.get("completion_tokens", reported_output_tokens)
                    )

                choices = body.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    delta_count += 1
    except Exception as error:
        return RequestSample(0.0, time.perf_counter() - started, 0, 0, False, type(error).__name__)

    finished = time.perf_counter()
    return RequestSample(
        ttft_seconds=(first_token_at or finished) - started,
        total_seconds=finished - started,
        output_tokens=reported_output_tokens or delta_count,
        input_tokens=input_tokens,
    )


async def run_level(
    base_url: str,
    model: str,
    prompt: str,
    concurrency: int,
    requests: int,
    max_output_tokens: int,
    label: str,
) -> BenchmarkResult:
    """Run one concurrency level to completion."""
    import httpx

    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:

        async def one() -> RequestSample:
            async with semaphore:
                return await stream_once(client, model, prompt, max_output_tokens)

        # Warm up so the first sample does not carry model load or cache-cold
        # cost into the measurement.
        await stream_once(client, model, prompt, 8)

        started = time.perf_counter()
        samples = await asyncio.gather(*(one() for _ in range(requests)))
        wall = time.perf_counter() - started

    result = summarise(label, concurrency, len(prompt) // 4, max_output_tokens, list(samples), wall)
    logger.info("benchmark_level_completed", **result.as_dict())
    return result


@dataclass
class BenchmarkSuite:
    """A full sweep across concurrency levels."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def print_table(self) -> None:
        """Print the latency/throughput curve."""
        print(f"\n{HEADER}")
        print("-" * len(HEADER))
        for result in self.results:
            print(result.summary_line())

    def as_dict(self) -> dict[str, Any]:
        return {"levels": [r.as_dict() for r in self.results]}
