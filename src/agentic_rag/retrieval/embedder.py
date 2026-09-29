"""BGE-M3 embedding with dense and learned-sparse output."""

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

EMBEDDINGS_FILENAME = "embeddings.npz"


@dataclass
class EmbeddingResult:
    """Dense and sparse representations for a batch of texts."""

    dense: list[list[float]]
    sparse: list[dict[int, float]]

    def __len__(self) -> int:
        return len(self.dense)


@dataclass
class Embedder:
    """Wraps BGE-M3, producing dense and sparse vectors in one forward pass.

    One model for both representations is the reason this project needs no
    separate sparse encoder: the lexical weights are learned alongside the
    dense vector rather than counted afterwards.
    """

    settings: Settings = field(default_factory=get_settings)

    @cached_property
    def _model(self) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        config = self.settings.embedding
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("embedding_model_loading", model=config.model_name, device=config.device)
        model = BGEM3FlagModel(
            config.model_name,
            use_fp16=config.use_fp16,
            device=config.device,
            cache_dir=str(config.cache_dir),
        )
        logger.info("embedding_model_loaded", model=config.model_name)
        return model

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts into dense vectors and sparse lexical weights."""
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])

        config = self.settings.embedding
        output = self._model.encode(
            texts,
            batch_size=config.batch_size,
            max_length=config.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return EmbeddingResult(
            dense=[vector.tolist() for vector in output["dense_vecs"]],
            sparse=[
                {int(token): float(weight) for token, weight in weights.items()}
                for weights in output["lexical_weights"]
            ],
        )

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        """Encode a single query, returning its dense and sparse vectors."""
        result = self.encode([text])
        return result.dense[0], result.sparse[0]


def save_embeddings(chunk_ids: list[str], result: EmbeddingResult, destination: Path) -> None:
    """Persist vectors to a compressed archive.

    Embedding the corpus is the most expensive step in the pipeline and its
    output never changes for a fixed corpus hash, so it is cached to disk and
    every later experiment reuses it.
    """
    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        chunk_ids=np.array(chunk_ids, dtype=object),
        dense=np.array(result.dense, dtype=np.float32),
        sparse=np.array([json.dumps(s) for s in result.sparse], dtype=object),
    )
    logger.info("embeddings_saved", path=str(destination), count=len(chunk_ids))


def load_embeddings(source: Path) -> tuple[list[str], EmbeddingResult]:
    """Load vectors from a compressed archive."""
    import numpy as np

    data = np.load(source, allow_pickle=True)
    chunk_ids = [str(cid) for cid in data["chunk_ids"]]
    result = EmbeddingResult(
        dense=[row.tolist() for row in data["dense"]],
        sparse=[{int(k): float(v) for k, v in json.loads(s).items()} for s in data["sparse"]],
    )
    logger.info("embeddings_loaded", path=str(source), count=len(chunk_ids))
    return chunk_ids, result
