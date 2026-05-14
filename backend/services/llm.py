"""
LLM provider abstraction — local-first.

Why local-first?
────────────────
The target environment for this system is on-prem, private-network, and
potentially air-gapped enterprise deployment. Cloud APIs (OpenAI, Bedrock)
are not assumed reachable. The default ``LLM_PROVIDER=ollama`` reflects
that constraint. Cloud providers are explicit opt-in via the
``LLM_PROVIDER`` environment variable.

Tiers
─────
  fast    → classification, grading, query rewriting, hallucination check
  smart   → final answer synthesis when the query needs multi-step reasoning
  rewrite → short query reformulation

Each tier maps to a model per provider — see ``_MODEL_MAP``. Tier choice is
orthogonal to provider choice: the same tier picks the right-sized model
for whichever provider is configured.

Provider resolution
───────────────────
  ``LLM_PROVIDER=ollama``  → Ollama (default, local)
  ``LLM_PROVIDER=openai``  → OpenAI
  ``LLM_PROVIDER=bedrock`` → AWS Bedrock
  ``LLM_PROVIDER=auto``    → first reachable of: ollama, openai, bedrock

In auto mode the chain is LOCAL FIRST. This is deliberate: the previous
``Bedrock → OpenAI → Ollama`` ordering treated local inference as the last
resort and signalled a cloud-first architecture. That ordering is wrong
for this product.
"""

import logging
import re
from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from core.config import get_settings
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

ModelTier = Literal["fast", "smart", "rewrite"]


# Per-provider model selection per tier. Customers can override via env vars.
_MODEL_MAP = {
    "ollama": {
        # The Ollama model is the same for every tier — local deployments
        # typically host one general-purpose instruct model. Differentiation
        # by tier would require multiple loaded models, which is wasteful
        # on a single GPU. Operators wanting smart-tier upgrades should run
        # a larger model (e.g. llama3.1:70b) and set LLM_MODEL accordingly.
        "fast":    None,   # filled from settings.llm_model at call time
        "smart":   None,
        "rewrite": None,
    },
    "openai": {
        "fast":    "gpt-4o-mini",
        "smart":   "gpt-4o",
        "rewrite": "gpt-4o-mini",
    },
    "bedrock": {
        "fast":    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "smart":   "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "rewrite": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    },
}

COSTS_PER_1M = {
    "haiku":     {"input": 1.0,   "output": 5.0},
    "sonnet":    {"input": 3.0,   "output": 15.0},
    "gpt4o_mini": {"input": 0.15, "output": 0.60},
}

_COMPLEX_PATTERNS = [
    r"\b(compare|contrast|versus|vs\.?)\b",
    r"\b(analyze|analyse|assess|evaluate)\b",
    r"\b(recommend|should i|what if|best option|optimal)\b",
    r"\b(risk|implication|consequence)\b",
    r"\band\b.{5,40}\b(also|then|after)\b",
]

_SIMPLE_PATTERNS = [
    r"^\s*what is\b",
    r"^\s*how much\b",
    r"^\s*how many\b",
    r"^\s*when\b",
    r"^\s*where\b",
    r"^\s*who\b",
    r"\b(minimum|maximum|requirement|fee|charge|rate)\b",
    r"\b(list|show|give me|tell me)\b",
]


def classify_query_complexity(query: str) -> ModelTier:
    """Route to ``smart`` only when the query genuinely needs it.

    Default is ``fast`` — retrieved context does the heavy lifting, not the
    model. Using the largest tier everywhere is the most common
    over-spending mistake in RAG systems.
    """
    q = query.lower()
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, q):
            logger.debug("Complex query → smart: %s", pattern)
            return "smart"
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, q):
            logger.debug("Simple query → fast: %s", pattern)
            return "fast"
    return "fast"  # default: cost-optimised


# ── Provider builders ────────────────────────────────────────────────────────

def _build_ollama(tier: ModelTier) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    settings = get_settings()
    logger.info("LLM %s → Ollama:%s", tier, settings.llm_model)
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    )


def _build_openai(tier: ModelTier) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    settings = get_settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY required for openai provider")
    model = _MODEL_MAP["openai"][tier]
    logger.info("LLM %s → OpenAI:%s", tier, model)
    return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0.1)


def _build_bedrock(tier: ModelTier) -> BaseChatModel:
    from langchain_aws import ChatBedrock
    settings = get_settings()
    model = _MODEL_MAP["bedrock"][tier]
    logger.info("LLM %s → Bedrock:%s", tier, model)
    return ChatBedrock(
        model_id=model,
        model_kwargs={"temperature": 0.1},
        region_name=settings.aws_region,
    )


_BUILDERS = {
    "ollama":  _build_ollama,
    "openai":  _build_openai,
    "bedrock": _build_bedrock,
}


# ── Public API ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=3)  # one cached instance per tier
def get_llm(tier: ModelTier = "fast") -> BaseChatModel:
    """Return a cached LLM instance for the given tier.

    The lru_cache means each tier is initialised at most once per process —
    no repeated client construction on the hot path. Settings are read
    inside (not passed as args) because Pydantic models are not hashable
    and cannot serve as cache keys.
    """
    settings = get_settings()
    provider = settings.resolve_llm_provider()
    return _BUILDERS[provider](tier)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Rough USD cost estimate for a single call.

    Returns 0 for local providers (Ollama) since there is no per-token
    monetary cost — though there is a real compute cost not modelled here.
    """
    costs = COSTS_PER_1M.get(model)
    if not costs:
        return 0.0
    return round(
        (input_tokens / 1_000_000) * costs["input"]
        + (output_tokens / 1_000_000) * costs["output"],
        6,
    )
