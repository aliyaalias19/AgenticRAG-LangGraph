"""
BM25 sparse retriever with safe on-disk caching.

Why BM25 alongside vector search
────────────────────────────────
Dense vectors excel at semantic similarity but can miss exact keyword
matches. BM25 is the opposite — strong on product codes, named entities,
account numbers, and precise terminology. Combining both (via RRF fusion
in ``retrieval/hybrid.py``) gives higher recall than either alone on
real-world enterprise documents.

Persistence: JSON, not pickle
─────────────────────────────
The previous implementation persisted the BM25 index with ``pickle``.
That is unsafe — ``pickle.load`` executes arbitrary Python code from the
file. In an enterprise deployment where the data volume might be mounted
from network storage or restored from a backup of unknown provenance,
this is a real attack surface. Loading a tampered ``bm25_persistent.pkl``
could give an attacker code execution as the API user.

This file persists the tokenised corpus and metadatas as JSON instead.
The ``BM25Okapi`` instance is re-built on load by calling its constructor
with the loaded tokens. JSON cannot execute code; the load surface is
limited to parser bugs (which are well-fuzzed in CPython's json module).

Tradeoff:
  • JSON is larger on disk than pickle (~2-3× for the same corpus).
  • Load is ~50ms slower for 10k chunks because ``BM25Okapi`` recomputes
    IDF stats from the tokenised corpus.
  • These costs are paid once per startup and are dwarfed by embedding
    model load time. The security trade is unambiguously worth it.

Stale-cache cleanup: an existing ``bm25_persistent.pkl`` from the prior
version is detected and ignored — ``load_from_disk`` falls back to
"index not found" and the next ingest rebuilds from scratch. We do not
auto-delete it; the operator should remove it explicitly.
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Bump this if the on-disk schema changes incompatibly so old caches are
# transparently ignored rather than mis-loaded.
_SCHEMA_VERSION = 1


def _tokenise(text: str) -> List[str]:
    """Match-the-build tokenisation used at both build and query time.

    Lowercase + whitespace split. Trivial by design — BM25's strength is
    inversely proportional to tokeniser cleverness on short queries. If
    you want stemming or stop-word removal, add it here and bump
    ``_SCHEMA_VERSION``.
    """
    return text.lower().split()


class BM25Retriever:
    """Thread-safe BM25 retriever with persistent index caching."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._index: Optional[BM25Okapi] = None
        self._texts: List[str] = []
        self._metadatas: List[dict] = []

    @property
    def _cache_path(self) -> Path:
        return Path(self._settings.data_dir) / "bm25_index.json"

    @property
    def _legacy_pickle_path(self) -> Path:
        return Path(self._settings.data_dir) / "bm25_persistent.pkl"

    def load_from_disk(self) -> bool:
        """Load persisted index. Returns True on success.

        If a legacy pickle cache exists alongside the JSON cache it is
        deliberately not loaded — the security trade-off (see module
        docstring) means we never deserialise untrusted Python.
        """
        if self._legacy_pickle_path.exists():
            logger.warning(
                "Ignoring legacy pickle cache %s — delete it manually after "
                "verifying the JSON index is rebuilt.",
                self._legacy_pickle_path,
            )

        if not self._cache_path.exists():
            return False
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read BM25 cache (%s); will rebuild", exc)
            return False

        if cached.get("schema_version") != _SCHEMA_VERSION:
            logger.warning(
                "BM25 cache schema mismatch (got %s, want %s); will rebuild",
                cached.get("schema_version"), _SCHEMA_VERSION,
            )
            return False

        texts = cached.get("texts", [])
        metadatas = cached.get("metadatas", [])
        tokenised = cached.get("tokens")

        if not texts:
            return False

        # Defensive: if tokens were not persisted (older write), recompute.
        if not tokenised:
            tokenised = [_tokenise(t) for t in texts]

        with self._lock:
            self._index = BM25Okapi(tokenised)
            self._texts = texts
            self._metadatas = metadatas

        logger.info("BM25 index loaded from disk", extra={"docs": len(texts)})
        return True

    def build(self, texts: List[str], metadatas: List[dict]) -> None:
        """(Re)build the index from raw texts and persist to disk."""
        logger.info("Building BM25 index", extra={"docs": len(texts)})
        tokenised = [_tokenise(t) for t in texts]
        index = BM25Okapi(tokenised)

        with self._lock:
            self._index = index
            self._texts = texts
            self._metadatas = metadatas

        Path(self._settings.data_dir).mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "texts": texts,
            "metadatas": metadatas,
            "tokens": tokenised,
        }
        # Atomic-ish write: write to a temp file then rename. Prevents a
        # half-written index from being loaded after a crash mid-write.
        tmp = self._cache_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(self._cache_path)
        logger.info("BM25 index persisted", extra={"path": str(self._cache_path)})

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> List[Document]:
        """Return top-k documents by BM25 score.

        ``where`` is an AND-of-equality filter on chunk metadata. For
        tenant isolation, pass ``{"tenant_id": ...}``.

        Implementation note: the BM25 index is shared across tenants in
        memory (a single ``BM25Okapi`` over the full corpus). To respect
        the filter without compromising recall we OVER-RETRIEVE by a
        factor of 5× top_k and post-filter the candidates. This works
        for small numbers of tenants; for unbounded-tenancy deployments
        the index should be partitioned per tenant (see
        ``BM25Retriever`` docstring for the upgrade path).
        """
        k = top_k or self._settings.bm25_top_k
        with self._lock:
            if self._index is None or not self._texts:
                return []
            scores = self._index.get_scores(_tokenise(query))

        # Over-retrieve: ask for 5× the requested k so post-filtering
        # rarely drops us below k results for an active tenant.
        candidate_count = max(k * 5, k)
        candidate_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:candidate_count]

        results: List[Document] = []
        for i in candidate_idx:
            if scores[i] <= 0:
                continue
            meta = self._metadatas[i] if i < len(self._metadatas) else {}
            if where and not all(meta.get(field) == value for field, value in where.items()):
                continue
            results.append(Document(
                page_content=self._texts[i],
                metadata={**meta, "bm25_score": float(scores[i])},
            ))
            if len(results) >= k:
                break
        return results

    def get_corpus_size(self) -> int:
        with self._lock:
            return len(self._texts)
