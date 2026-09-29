"""BGE-M3 embedding with dense and sparse output."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Dense and sparse representations for a batch of texts."""

    dense: list[list[float]]
    sparse: list[dict[int, float]]

    def __len__(self) -> int:
        return len(self.dense)


@dataclass
class Embedder:
    """Wraps BGE-M3, producing dense and sparse vectors in one pass."""

    settings: Settings = field(default_factory=get_settings)

    @cached_property
    def _model(self) -> Any:
        from FlagEmbedding import BGEM3FlagModel

        config = self.settings.embedding
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "embedding_model_loading",
            model=config.model_name,
            device=config.device,
        )
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

        dense = [vector.tolist() for vector in output["dense_vecs"]]
        sparse = [
            {int(token_id): float(weight) for token_id, weight in weights.items()}
            for weights in output["lexical_weights"]
        ]
        return EmbeddingResult(dense=dense, sparse=sparse)

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        """Encode a single query, returning its dense and sparse vectors."""
        result = self.encode([text])
        return result.dense[0], result.sparse[0]
