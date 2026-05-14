"""
Embedding provider abstraction — decoupled from the LLM provider.

CRITICAL invariant
──────────────────
The embedding model used at ingest time MUST match the one used at query
time. Mixing providers (OpenAI at ingest, Ollama at query) produces
vectors in incompatible spaces and similarity search silently returns
noise. ``get_embeddings()`` is cached for the process lifetime to make
this invariant easy to hold.

Why decoupled from LLM
──────────────────────
Enterprise customers commonly want:
  • Local embeddings (data sovereignty, zero egress) + cloud LLM, or
  • Cloud embeddings + local LLM during evaluation.

The previous ``use_openai`` flag forced both to the same provider. The
two are now controlled independently via ``LLM_PROVIDER`` and
``EMBEDDING_PROVIDER``.

Default
───────
``EMBEDDING_PROVIDER=ollama``. Local-first, matches the deployment story.
"""

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from core.config import get_settings
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _build_ollama() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings
    settings = get_settings()
    logger.info("Embeddings → Ollama:%s", settings.embed_model)
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def _build_openai() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    settings = get_settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY required for openai embeddings")
    logger.info("Embeddings → OpenAI:%s", settings.openai_embed_model)
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )


def _build_bedrock() -> Embeddings:
    from langchain_aws import BedrockEmbeddings
    settings = get_settings()
    logger.info("Embeddings → Bedrock:%s", settings.bedrock_embed_model)
    return BedrockEmbeddings(
        model_id=settings.bedrock_embed_model,
        region_name=settings.aws_default_region,
    )


_BUILDERS = {
    "ollama":  _build_ollama,
    "openai":  _build_openai,
    "bedrock": _build_bedrock,
}


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the cached embeddings instance for the configured provider.

    Cached because (a) the model load is expensive — a single API client
    or HTTP session is reused — and (b) the invariant above requires every
    call site to see the same provider for the process lifetime.
    """
    settings = get_settings()
    provider = settings.resolve_embedding_provider()
    return _BUILDERS[provider]()
