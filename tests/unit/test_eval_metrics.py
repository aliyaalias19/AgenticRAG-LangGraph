"""Tests for retrieval and answer quality metrics."""

import pytest

from agentic_rag.eval.metrics import (
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from agentic_rag.eval.quality import (
    QualityAggregate,
    QualityScores,
    ShippingThreshold,
    context_precision,
    context_recall,
)


class TestRecall:
    def test_all_gold_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 10) == 1.0

    def test_half_gold_found(self) -> None:
        assert recall_at_k(["a", "x"], ["a", "b"], 10) == 0.5

    def test_cutoff_is_applied(self) -> None:
        assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 3) == 1.0

    def test_empty_gold_is_zero(self) -> None:
        assert recall_at_k(["a"], [], 10) == 0.0


class TestPrecision:
    def test_all_returned_are_gold(self) -> None:
        assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_half_returned_are_gold(self) -> None:
        assert precision_at_k(["a", "x"], ["a"], 2) == 0.5

    def test_empty_retrieval_is_zero(self) -> None:
        assert precision_at_k([], ["a"], 5) == 0.0


class TestRankSensitivity:
    def test_reciprocal_rank_rewards_early_hits(self) -> None:
        assert reciprocal_rank(["a", "x"], ["a"]) == 1.0
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_ndcg_prefers_earlier_gold(self) -> None:
        early = ndcg_at_k(["a", "x", "y"], ["a"], 10)
        late = ndcg_at_k(["x", "y", "a"], ["a"], 10)
        assert early > late
        assert early == 1.0

    def test_ndcg_is_bounded(self) -> None:
        assert 0.0 <= ndcg_at_k(["x", "a", "y"], ["a", "y"], 10) <= 1.0

    def test_average_precision_rewards_clustering_at_the_top(self) -> None:
        top = average_precision(["a", "b", "x", "y"], ["a", "b"])
        spread = average_precision(["a", "x", "y", "b"], ["a", "b"])
        assert top > spread

    def test_hit_at_k_is_binary(self) -> None:
        assert hit_at_k(["x", "a"], ["a"], 10) == 1.0
        assert hit_at_k(["x", "y"], ["a"], 10) == 0.0


class TestContextMetrics:
    def test_context_precision_is_rank_weighted(self) -> None:
        assert context_precision(["a", "x", "y"], ["a"]) > context_precision(["x", "y", "a"], ["a"])

    def test_context_recall_counts_gold_found(self) -> None:
        assert context_recall(["a", "b"], ["a", "b"]) == 1.0
        assert context_recall(["a"], ["a", "b"]) == 0.5


class TestShippingThreshold:
    def test_all_metrics_must_clear(self) -> None:
        threshold = ShippingThreshold()
        assert threshold.passes(QualityScores(0.7, 0.8, 0.9, 0.85))

    def test_one_weak_metric_blocks_shipping(self) -> None:
        threshold = ShippingThreshold()
        scores = QualityScores(0.99, 0.99, 0.40, 0.99)
        assert not threshold.passes(scores)
        assert set(threshold.failures(scores)) == {"faithfulness"}

    def test_failures_report_observed_and_required(self) -> None:
        threshold = ShippingThreshold(faithfulness=0.85)
        observed, required = threshold.failures(QualityScores(0.9, 0.9, 0.5, 0.9))["faithfulness"]
        assert observed == 0.5
        assert required == 0.85

    def test_harmonic_mean_punishes_a_weak_metric(self) -> None:
        balanced = QualityScores(0.8, 0.8, 0.8, 0.8)
        lopsided = QualityScores(1.0, 1.0, 0.2, 1.0)
        assert balanced.harmonic_mean > lopsided.harmonic_mean

    def test_a_zero_metric_zeroes_the_harmonic_mean(self) -> None:
        assert QualityScores(1.0, 1.0, 0.0, 1.0).harmonic_mean == 0.0


class TestAggregate:
    def test_mean_is_element_wise(self) -> None:
        aggregate = QualityAggregate()
        aggregate.add(QualityScores(0.6, 0.6, 0.6, 0.6))
        aggregate.add(QualityScores(0.8, 0.8, 0.8, 0.8))
        assert aggregate.mean().faithfulness == pytest.approx(0.7)

    def test_empty_aggregate_is_zero(self) -> None:
        assert QualityAggregate().mean().faithfulness == 0.0

    def test_report_states_the_verdict(self) -> None:
        aggregate = QualityAggregate()
        aggregate.add(QualityScores(0.9, 0.9, 0.9, 0.9))
        report = aggregate.report()
        assert report["ships"] is True
        assert report["questions"] == 1
        assert report["failures"] == {}
