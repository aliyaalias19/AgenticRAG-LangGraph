"""Persisted BM25 lexical index with CJK-aware tokenization.

BM25 scores a query against a document by term frequency, damped so that
repeated terms give diminishing returns, and weighted by inverse document
frequency so that rare terms count for more. Unlike a dense embedding it
cannot match paraphrases, but it is exact on identifiers, flags and command
names -- which is precisely where dense retrieval is weakest.
"""

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agentic_rag.obs.logging import get_logger
from agentic_rag.retrieval.store import SearchHit


class ChunkLike(Protocol):
    """Anything with the fields BM25 needs.

    Structural rather than nominal so retrieval never imports ingest: the
    two packages stay independently testable.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    language: str
    heading_path: list[str]
    section_path: list[str]
    source_path: str


logger = get_logger(__name__)

INDEX_FILENAME = "bm25_index.pkl"

# Latin words, numbers, and single CJK characters. Chinese has no spaces, so
# whitespace tokenization silently produces one giant token per sentence;
# indexing each character instead gives usable unigram matching.
TOKEN_PATTERN = re.compile(r"[a-z0-9_.-]+|[一-鿿]")

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
        "for",
        "from",
        "has",
        "have",
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
        "that",
        "the",
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
    ]
)

K1 = 1.5
B = 0.75


# Suffixes stripped in order, longest first. This is deliberately lighter than
# a full Porter stemmer. Documentation queries need "drain" to match
# "draining" and "volume" to match "volumes"; anything more aggressive starts
# merging distinct Kubernetes terms.
SUFFIXES: tuple[tuple[str, int], ...] = (
    ("ingly", 5),
    ("edly", 4),
    ("ing", 3),
    ("ies", 3),
    ("ed", 2),
    ("es", 2),
    ("s", 1),
)

MIN_TOKEN_LENGTH = 3
MIN_STEM_LENGTH = 3
# Doubled finals that are part of the word rather than an artefact of
# suffixing, so "call" must not become "cal".
KEEP_DOUBLED = frozenset("sl")


def stem(token: str) -> str:
    """Return a lightly stemmed form of ``token``.

    Only alphabetic tokens longer than ``MIN_TOKEN_LENGTH`` are stemmed, so
    identifiers, flags, version strings and CJK characters pass through
    untouched.

    The final step strips a trailing "e" so that the inflected and base forms
    converge on the same stem: without it "volumes" stems to "volum" while
    "volume" stays whole, and a query never matches its own plural.
    """
    if len(token) <= MIN_TOKEN_LENGTH or not token.isalpha():
        return token

    for suffix, length in SUFFIXES:
        if token.endswith(suffix) and len(token) - length >= MIN_STEM_LENGTH:
            token = token[:-length]
            if suffix == "ies":
                token += "y"
            break

    if len(token) > MIN_STEM_LENGTH and token[-1] == token[-2] and token[-1] not in KEEP_DOUBLED:
        token = token[:-1]

    if len(token) > MIN_STEM_LENGTH and token.endswith("e"):
        token = token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    """Return stemmed BM25 tokens for ``text``."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS]


@dataclass
class BM25Document:
    """Metadata retained for a single indexed chunk."""

    chunk_id: str
    doc_id: str
    relative_id: str
    doc_title: str
    heading_path: tuple[str, ...]
    content: str
    language: str
    section: str
    tenant: str
    source_path: str


@dataclass
class BM25Index:
    """An in-memory BM25 index that can be persisted to disk."""

    documents: list[BM25Document] = field(default_factory=list)
    term_frequencies: list[dict[str, int]] = field(default_factory=list)
    document_lengths: list[int] = field(default_factory=list)
    document_frequency: dict[str, int] = field(default_factory=dict)
    postings: dict[str, list[int]] = field(default_factory=dict)
    average_length: float = 0.0

    @property
    def size(self) -> int:
        return len(self.documents)

    def add(self, document: BM25Document) -> None:
        """Add one document to the index."""
        tokens = tokenize(
            f"{document.doc_title} {' '.join(document.heading_path)} {document.content}"
        )
        counts = Counter(tokens)
        position = len(self.documents)

        self.documents.append(document)
        self.term_frequencies.append(dict(counts))
        self.document_lengths.append(len(tokens))

        for term in counts:
            self.document_frequency[term] = self.document_frequency.get(term, 0) + 1
            self.postings.setdefault(term, []).append(position)

    def finalise(self) -> None:
        """Compute corpus statistics after all documents are added."""
        total = sum(self.document_lengths)
        self.average_length = total / len(self.documents) if self.documents else 0.0
        logger.info(
            "bm25_index_built",
            documents=len(self.documents),
            vocabulary=len(self.document_frequency),
            average_length=round(self.average_length, 1),
        )

    def _idf(self, term: str) -> float:
        """Return the inverse document frequency of ``term``.

        Uses the BM25+ variant of the Robertson IDF, whose ``+ 1`` inside the
        logarithm keeps the value non-negative for terms that appear in more
        than half the corpus.
        """
        frequency = self.document_frequency.get(term, 0)
        if frequency == 0:
            return 0.0
        n = len(self.documents)
        return math.log(1 + (n - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        query: str,
        limit: int,
        tenant: str | None = "public",
        language: str | None = None,
        sections: tuple[str, ...] = (),
    ) -> list[SearchHit]:
        """Return the top ``limit`` documents for ``query``."""
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []

        scores: dict[int, float] = {}
        for term in set(query_terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for position in self.postings.get(term, ()):
                frequency = self.term_frequencies[position].get(term, 0)
                length = self.document_lengths[position]
                average = self.average_length or 1.0
                norm = 1 - B + B * (length / average)
                contribution = idf * (frequency * (K1 + 1)) / (frequency + K1 * norm)
                scores[position] = scores.get(position, 0.0) + contribution

        filtered = [
            (position, score)
            for position, score in scores.items()
            if self._matches(self.documents[position], tenant, language, sections)
        ]
        filtered.sort(key=lambda item: (-item[1], self.documents[item[0]].chunk_id))

        return [
            SearchHit(
                chunk_id=self.documents[position].chunk_id,
                score=score,
                doc_id=self.documents[position].doc_id,
                relative_id=self.documents[position].relative_id,
                doc_title=self.documents[position].doc_title,
                heading_path=self.documents[position].heading_path,
                content=self.documents[position].content,
                language=self.documents[position].language,
                section=self.documents[position].section,
                source_path=self.documents[position].source_path,
            )
            for position, score in filtered[:limit]
        ]

    @staticmethod
    def _matches(
        document: BM25Document,
        tenant: str | None,
        language: str | None,
        sections: tuple[str, ...],
    ) -> bool:
        """Return True when the document satisfies the access filters."""
        if tenant is not None and document.tenant != tenant:
            return False
        if language is not None and document.language != language:
            return False
        return not (sections and document.section not in sections)

    def save(self, destination: Path) -> None:
        """Persist the index to disk."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("bm25_index_saved", path=str(destination), documents=self.size)

    @staticmethod
    def load(source: Path) -> "BM25Index":
        """Load a persisted index from disk."""
        with source.open("rb") as handle:
            index: BM25Index = pickle.load(handle)  # noqa: S301 - own artifact
        logger.info("bm25_index_loaded", path=str(source), documents=index.size)
        return index


def build_index(chunks: list[ChunkLike], tenant: str = "public") -> BM25Index:
    """Build a BM25 index from a list of chunks."""
    index = BM25Index()
    for chunk in chunks:
        index.add(
            BM25Document(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                relative_id=getattr(chunk, "relative_id", ""),
                doc_title=chunk.doc_title,
                heading_path=tuple(chunk.heading_path),
                content=chunk.content,
                language=chunk.language,
                section=chunk.section_path[0] if chunk.section_path else "other",
                tenant=tenant,
                source_path=chunk.source_path,
            )
        )
    index.finalise()
    return index
