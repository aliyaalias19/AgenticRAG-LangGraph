"""Stratified sampling of documents for question generation."""

import random
from collections import defaultdict

from agentic_rag.ingest.models import Document
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


def _top_section(document: Document) -> str:
    return document.section_path[0] if document.section_path else "other"


def allocate_targets(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Distribute ``total`` across sections using the largest-remainder method.

    Guarantees the allocations sum to exactly ``total`` when weights sum to 1.
    """
    exact = {section: total * weight for section, weight in weights.items()}
    allocated = {section: int(value) for section, value in exact.items()}

    remaining = total - sum(allocated.values())
    by_remainder = sorted(
        exact,
        key=lambda section: (-(exact[section] - allocated[section]), section),
    )

    for section in by_remainder[:remaining]:
        allocated[section] += 1

    return allocated


def sample_documents(
    documents: list[Document],
    *,
    total: int,
    section_weights: dict[str, float],
    min_chars: int,
    max_chars: int,
    seed: int,
    language: str = "en",
) -> list[Document]:
    """Return a stratified sample of documents suitable for question generation."""
    eligible = [
        doc
        for doc in documents
        if doc.language == language and min_chars <= doc.char_count <= max_chars
    ]

    by_section: dict[str, list[Document]] = defaultdict(list)
    for doc in eligible:
        by_section[_top_section(doc)].append(doc)

    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    targets = allocate_targets(total, section_weights)
    sampled: list[Document] = []

    for section in sorted(targets):
        pool = sorted(by_section.get(section, []), key=lambda d: d.doc_id)
        target = targets[section]

        if not pool:
            logger.warning("sampling_section_empty", section=section)
            continue

        take = min(target, len(pool))
        if take < target:
            logger.warning(
                "sampling_section_short",
                section=section,
                requested=target,
                available=len(pool),
            )
        sampled.extend(rng.sample(pool, take))

    sampled.sort(key=lambda d: d.doc_id)

    logger.info(
        "documents_sampled",
        requested=total,
        sampled=len(sampled),
        eligible=len(eligible),
    )
    return sampled
