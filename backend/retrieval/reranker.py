"""
Cross-encoder reranker.

Why reranking:
  Both BM25 and vector search rank by independent single-query scores.
  A cross-encoder jointly encodes the (query, passage) pair, giving a
  relevance score that explicitly models their interaction. This typically
  improves precision at k=1-3 by 5-15% compared to bi-encoder retrieval alone.

Why ms-marco-MiniLM:
  Fast enough for interactive use (~50ms for 5 candidates on CPU), trained on
  MS MARCO passage ranking, and freely available. For production at scale,
  a GPU-hosted model or a hosted API reranker (Cohere, Jina) would replace it.

Lazy loading:
  sentence-transformers downloads ~80MB on first use. Lazy loading avoids
  blocking startup if the model isn't cached yet.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_id: str = _MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._available: Optional[bool] = None

    def _load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_id)
            self._available = True
            logger.info("Cross-encoder loaded: %s", self._model_id)
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s) — reranking disabled", exc)
            self._available = False
        return self._available

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Return docs sorted by cross-encoder score (descending).

        Returns original order if the model failed to load or len(docs) <= 1.
        Attaches 'rerank_score' to each document's metadata.
        """
        if len(docs) <= 1 or not self._load() or self._model is None:
            return docs
        try:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            result = []
            for score, doc in ranked:
                doc.metadata["rerank_score"] = float(score)
                result.append(doc)
            return result
        except Exception as exc:
            logger.warning("Reranking failed: %s — returning original order", exc)
            return docs
