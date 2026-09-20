"""Lexical overlap measurement between questions and their gold passages."""

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "my",
        "me",
        "does't",
        "don't",
        "not",
        "no",
    ]
)


def tokenize(text: str) -> set[str]:
    """Return the set of content tokens in ``text``."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def overlap_ratio(question: str, passages: list[str]) -> float:
    """Return the fraction of the question's content tokens found in the passages.

    A value near 1.0 means the question reuses the passage vocabulary, making
    retrieval trivially easy. Near 0.0 means retrieval must match semantics.
    """
    question_tokens = tokenize(question)
    if not question_tokens:
        return 0.0

    passage_tokens: set[str] = set()
    for passage in passages:
        passage_tokens |= tokenize(passage)

    return len(question_tokens & passage_tokens) / len(question_tokens)
