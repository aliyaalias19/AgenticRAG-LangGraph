"""Health endpoint. Metrics are served from /metrics in Prometheus format."""

import logging

import httpx
from fastapi import APIRouter

from core.config import get_settings
from ingestion.pipeline import list_ingested_files
from models.responses import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Observability"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check():
    """Returns connectivity status, ingested document list, and feature flags."""
    ollama_ok = False
    ollama_models = []

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                ollama_models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass

    # Surface the resolved provider chain so operators can verify the
    # local-first defaults are in effect without reading env vars.
    try:
        llm_resolved = settings.resolve_llm_provider()
    except Exception:
        llm_resolved = "unavailable"
    try:
        emb_resolved = settings.resolve_embedding_provider()
    except Exception:
        emb_resolved = "unavailable"

    return HealthResponse(
        status="ok",
        version=settings.api_version,
        ollama_connected=ollama_ok,
        ollama_models=ollama_models,
        ingested_files=list_ingested_files(settings),
        features=[
            "local_first_provider_chain",
            "structured_json_logging",
            "request_correlation_ids",
            "fail_fast_config_validation",
            "langgraph_state_machine",
            "hybrid_retrieval_rrf",
            "cross_encoder_reranking",
            "heuristic_grading_with_llm_fallback",
            "hallucination_detection",
            "confidence_scoring",
            "sentence_level_citations",
            "sse_streaming",
            "ragas_evaluation",
            "sha256_deduplication",
            f"llm_provider={llm_resolved}",
            f"embedding_provider={emb_resolved}",
        ],
    )


