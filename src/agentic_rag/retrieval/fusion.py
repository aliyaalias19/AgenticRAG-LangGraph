"""Reciprocal Rank Fusion for combining ranked retrieval results."""

from dataclasses import replace

from agentic_rag.retrieval.store import SearchHit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]],
    *,
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked result lists by reciprocal rank.

    Each result contributes ``weight / (k + rank)`` to its chunk's score, where
    rank is 1-based. RRF is used rather than score normalisation because dense
    cosine similarities and BM25 scores live on incomparable scales; only their
    orderings carry comparable information.

    ``k`` damps the influence of the very top ranks. Lower values trust each
    list's first result more aggressively.

    One property is worth knowing before reading the scores: because ``1/x``
    is convex, ``1/(k+1) + 1/(k+3)`` exceeds ``2/(k+2)``. A passage ranked
    first by one retriever and third by another therefore outranks a passage
    both retrievers put second. RRF rewards a strong peak rank, not
    consistency -- which is what you want when the two retrievers fail on
    different query types, and worth remembering when a "compromise" result
    does not surface where intuition says it should.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        message = "weights length must match rankings length"
        raise ValueError(message)

    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            best.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused = [replace(best[chunk_id], score=score) for chunk_id, score in ordered]
    return fused[:limit] if limit else fused
