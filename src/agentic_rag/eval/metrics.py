"""Retrieval quality metrics.

Recall answers "did we find the evidence", precision answers "was what we
returned worth reading", and the rank-sensitive metrics answer "how far down
did the reader have to look". A retrieval change that improves recall while
destroying MRR has made the system worse for a person, so all three are
reported together.
"""

import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of gold chunks appearing in the top ``k`` retrieved."""
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved that are gold."""
    top = retrieved[:k]
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(1 for chunk_id in top if chunk_id in gold_set) / len(top)


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1.0 when any gold chunk appears in the top ``k``, else 0.0."""
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    """Reciprocal of the rank of the first gold chunk."""
    gold_set = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not gold:
        return 0.0
    gold_set = set(gold)

    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0


def average_precision(retrieved: list[str], gold: list[str]) -> float:
    """Mean of the precisions measured at each gold hit."""
    if not gold:
        return 0.0
    gold_set = set(gold)
    hits = 0
    total = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            hits += 1
            total += hits / rank
    return total / len(gold_set)
