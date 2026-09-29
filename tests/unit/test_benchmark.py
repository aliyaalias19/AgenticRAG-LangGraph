"""Tests for benchmark aggregation."""

from agentic_rag.serving.benchmark import RequestSample, percentile, summarise


def sample(ttft: float, total: float, tokens: int = 100) -> RequestSample:
    return RequestSample(
        ttft_seconds=ttft, total_seconds=total, output_tokens=tokens, input_tokens=500
    )


class TestPercentile:
    def test_nearest_rank_returns_an_observed_value(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0.5) in values
        assert percentile(values, 0.95) in values

    def test_p50_is_the_middle(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p99_is_near_the_top(self) -> None:
        assert percentile([float(i) for i in range(100)], 0.99) == 99.0

    def test_empty_is_zero(self) -> None:
        assert percentile([], 0.95) == 0.0

    def test_unsorted_input_is_handled(self) -> None:
        assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0


class TestSampleDerivations:
    def test_inter_token_latency_excludes_the_first_token(self) -> None:
        s = sample(ttft=0.5, total=1.5, tokens=11)
        assert s.inter_token_seconds == 0.1

    def test_single_token_has_no_inter_token_latency(self) -> None:
        assert sample(0.5, 0.6, tokens=1).inter_token_seconds == 0.0

    def test_decode_rate_excludes_prefill(self) -> None:
        s = sample(ttft=1.0, total=2.0, tokens=51)
        assert s.output_tokens_per_second == 50.0

    def test_zero_decode_time_is_handled(self) -> None:
        assert sample(1.0, 1.0, tokens=10).output_tokens_per_second == 0.0


class TestSummarise:
    def test_failures_are_counted_but_excluded_from_latency(self) -> None:
        samples = [sample(0.3, 2.0) for _ in range(10)]
        samples.append(RequestSample(0.0, 0.0, 0, 0, False, "Timeout"))
        result = summarise("x", 4, 500, 128, samples, wall_seconds=5.0)

        assert result.requests == 11
        assert result.failures == 1
        assert result.ttft_p50 == 0.3

    def test_aggregate_throughput_uses_wall_time(self) -> None:
        samples = [sample(0.3, 2.0, tokens=100) for _ in range(10)]
        result = summarise("x", 4, 500, 128, samples, wall_seconds=10.0)
        assert result.aggregate_output_tokens_per_second == 100.0

    def test_per_stream_rate_is_independent_of_concurrency(self) -> None:
        samples = [sample(0.5, 2.5, tokens=101) for _ in range(8)]
        low = summarise("a", 1, 500, 128, samples, wall_seconds=20.0)
        high = summarise("b", 16, 500, 128, samples, wall_seconds=2.5)

        assert low.output_tokens_per_second_per_stream == (high.output_tokens_per_second_per_stream)
        assert high.aggregate_output_tokens_per_second > (low.aggregate_output_tokens_per_second)

    def test_tail_latency_is_visible(self) -> None:
        """A mean would hide the slow tail; percentiles must not."""
        samples = [sample(0.2, 1.0) for _ in range(95)]
        samples += [sample(4.0, 6.0) for _ in range(5)]
        result = summarise("x", 8, 500, 128, samples, wall_seconds=10.0)

        assert result.ttft_p50 == 0.2
        assert result.ttft_p99 == 4.0

    def test_all_failures_produce_zeros_not_errors(self) -> None:
        samples = [RequestSample(0.0, 1.0, 0, 0, False, "Timeout") for _ in range(5)]
        result = summarise("x", 4, 500, 128, samples, wall_seconds=5.0)

        assert result.failures == 5
        assert result.ttft_p50 == 0.0
        assert result.aggregate_output_tokens_per_second == 0.0

    def test_summary_line_has_one_column_per_metric(self) -> None:
        result = summarise("x", 4, 500, 128, [sample(0.3, 2.0)], wall_seconds=2.0)
        # concurrency, TTFT p50/p95/p99, ITL p50, per-stream, aggregate
        assert len(result.summary_line().split()) == 7
