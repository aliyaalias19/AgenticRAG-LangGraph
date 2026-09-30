"""Cross-lingual retrieval evaluation over parallel documentation.

Documents that exist in more than one language under the same ``relative_id``
are hand-aligned parallel data: a passage and its translation are, by
construction, about the same thing. That makes a passage in one language a
legitimate query whose correct answer is the document carrying the same
``relative_id`` in the other -- ground truth at no labelling cost.

This module holds the pure parts of that evaluation. The retrieval calls
themselves live in ``scripts/eval_crosslingual.py``.
"""

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_QUERY_CHARS = 400


@dataclass(frozen=True)
class DocumentPair:
    """One document available in two languages."""

    relative_id: str
    first: dict[str, Any]
    second: dict[str, Any]


def load_pairs(chunks_path: Path, languages: tuple[str, str] = ("en", "zh")) -> list[DocumentPair]:
    """Return documents present in both languages, ordered by relative_id.

    Only the first chunk of each document is kept: the query is a passage, not
    the whole document, and taking a consistent position avoids favouring
    documents that happen to be longer.
    """
    rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    first_chunk: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["relative_id"], row["language"])
        if key not in first_chunk:
            first_chunk[key] = row

    seen: dict[str, set[str]] = collections.defaultdict(set)
    for relative_id, language in first_chunk:
        seen[relative_id].add(language)

    wanted = set(languages)
    pairs = [
        DocumentPair(
            relative_id=relative_id,
            first=first_chunk[(relative_id, languages[0])],
            second=first_chunk[(relative_id, languages[1])],
        )
        for relative_id, present in seen.items()
        if wanted <= present
    ]
    pairs.sort(key=lambda pair: pair.relative_id)
    return pairs


def make_query(
    chunk: dict[str, Any],
    chars: int = DEFAULT_QUERY_CHARS,
    use_title: bool = True,
) -> str:
    """Return the query text drawn from ``chunk``.

    Titles are translated near-literally, so including one makes the task
    substantially easier and can put every method on the ceiling. Excluding it
    measures whether the body text alone carries enough signal.
    """
    text = f"{chunk['doc_title']}\n{chunk['content']}" if use_title else chunk["content"]
    return text[:chars]


def rank_of(payloads: list[dict[str, Any]], relative_id: str) -> int | None:
    """Return the 1-based rank of ``relative_id``, or None if it is absent."""
    for position, payload in enumerate(payloads, start=1):
        if payload.get("relative_id") == relative_id:
            return position
    return None


def summarise(ranks: list[int | None], total: int) -> dict[str, float]:
    """Return recall at several cutoffs plus MRR.

    Misses count against the total rather than being dropped, so the figures
    describe the whole query set and not just the queries that succeeded.
    """
    if total <= 0:
        return {"queries": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}

    found = [rank for rank in ranks if rank is not None]
    return {
        "queries": total,
        "recall@1": round(sum(1 for r in found if r <= 1) / total, 4),
        "recall@5": round(sum(1 for r in found if r <= 5) / total, 4),
        "recall@10": round(sum(1 for r in found if r <= 10) / total, 4),
        "mrr": round(sum(1.0 / r for r in found) / total, 4),
    }
