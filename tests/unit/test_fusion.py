"""Tests for reciprocal rank fusion."""

import pytest

from agentic_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_rag.retrieval.store import SearchHit


def hits(*ids: str) -> list[SearchHit]:
    return [
        SearchHit(chunk_id=i, score=1.0 / rank, content=f"body {i}")
        for rank, i in enumerate(ids, start=1)
    ]


class TestFusion:
    def test_single_ranking_preserves_order(self) -> None:
        fused = reciprocal_rank_fusion([hits("a", "b", "c")])
        assert [h.chunk_id for h in fused] == ["a", "b", "c"]

    def test_appearing_in_both_lists_beats_appearing_in_one(self) -> None:
        dense = hits("shared", "dense_only")
        lexical = hits("lexical_only", "shared")
        fused = reciprocal_rank_fusion([dense, lexical])
        assert fused[0].chunk_id == "shared"

    def test_a_strong_peak_rank_beats_consistent_middling_ranks(self) -> None:
        """1/(k+1) + 1/(k+3) > 2/(k+2): the reciprocal is convex."""
        fused = reciprocal_rank_fusion([hits("a", "b", "c"), hits("c", "b", "a")])
        scores = {h.chunk_id: h.score for h in fused}
        assert scores["a"] > scores["b"]
        assert scores["c"] > scores["b"]
        assert scores["a"] == pytest.approx(scores["c"])

    def test_top_of_both_lists_outranks_top_of_one(self) -> None:
        fused = reciprocal_rank_fusion([hits("x", "a"), hits("x", "b")])
        assert fused[0].chunk_id == "x"

    def test_incomparable_scales_do_not_matter(self) -> None:
        """Dense cosine and BM25 scores differ by orders of magnitude."""
        dense = [SearchHit(chunk_id="a", score=0.81), SearchHit(chunk_id="b", score=0.79)]
        lexical = [SearchHit(chunk_id="b", score=41.2), SearchHit(chunk_id="a", score=3.1)]
        fused = reciprocal_rank_fusion([dense, lexical])
        assert {h.chunk_id for h in fused} == {"a", "b"}
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_weights_reorder_signal_specific_results(self) -> None:
        dense = hits("d1", "shared")
        lexical = hits("l1", "shared")

        dense_heavy = [
            h.chunk_id for h in reciprocal_rank_fusion([dense, lexical], weights=[5.0, 1.0])
        ]
        lexical_heavy = [
            h.chunk_id for h in reciprocal_rank_fusion([dense, lexical], weights=[1.0, 5.0])
        ]

        assert dense_heavy.index("d1") < dense_heavy.index("l1")
        assert lexical_heavy.index("l1") < lexical_heavy.index("d1")

    def test_lower_k_sharpens_the_top_rank(self) -> None:
        rankings = [hits("a", "b", "c"), hits("b", "c", "a")]
        sharp = reciprocal_rank_fusion(rankings, k=1)
        flat = reciprocal_rank_fusion(rankings, k=1000)
        spread = lambda f: f[0].score - f[-1].score  # noqa: E731
        assert spread(sharp) > spread(flat)

    def test_deduplicates_across_lists(self) -> None:
        fused = reciprocal_rank_fusion([hits("a", "b"), hits("a", "b")])
        assert len(fused) == 2

    def test_payload_is_preserved(self) -> None:
        fused = reciprocal_rank_fusion([hits("a")])
        assert fused[0].content == "body a"

    def test_limit_truncates(self) -> None:
        assert len(reciprocal_rank_fusion([hits(*"abcdef")], limit=3)) == 3

    def test_empty_rankings_return_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_mismatched_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            reciprocal_rank_fusion([hits("a"), hits("b")], weights=[1.0])

    def test_is_deterministic_under_ties(self) -> None:
        rankings = [hits("b", "a"), hits("a", "b")]
        first = [h.chunk_id for h in reciprocal_rank_fusion(rankings)]
        second = [h.chunk_id for h in reciprocal_rank_fusion(rankings)]
        assert first == second == sorted(first)
