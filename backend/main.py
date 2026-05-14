"""
Agentic RAG API — entry point.

This file is intentionally thin. All business logic lives in dedicated modules:
  agents/      LangGraph pipeline
  retrieval/   Hybrid BM25 + vector retrieval
  ingestion/   PDF ingestion pipeline
  evaluation/  RAGAS quality evaluation
  api/routes/  FastAPI route handlers
  core/        Configuration, exceptions, and structured logging
  services/    Provider-abstracted LLM and embedding clients
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import documents, evaluation, health, query
from core.config import get_settings
from core.exceptions import ConfigurationError
from core.logging import set_request_id, setup_logging
from infrastructure.prometheus import (
    AUTH_REJECTIONS,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    render_metrics,
)

settings = get_settings()
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting Agentic RAG API", extra={
        "api_version": settings.api_version,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    })

    try:
        warnings = settings.validate_at_startup()
        for w in warnings:
            logger.warning("Config check: %s", w)
    except ConfigurationError as exc:
        logger.error("Startup configuration error: %s", exc)
        raise

    try:
        from agents.graph import get_rag_graph
        get_rag_graph()
        logger.info("LangGraph pipeline pre-compiled")
    except Exception as exc:
        logger.warning("LangGraph pre-warm failed (non-fatal)", extra={"err": str(exc)})

    try:
        from retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(settings)
        bm25.load_from_disk()
    except Exception as exc:
        logger.warning("BM25 pre-load failed (non-fatal)", extra={"err": str(exc)})

    # Pre-warm the cross-encoder. First call to ``CrossEncoder(model_id)``
    # downloads ~570MB from HuggingFace and loads it into RAM. Doing it here
    # moves that cost from the user's first query (where it caused 4-minute
    # waits) to container boot. With the hf_cache volume in compose, the
    # download is a one-time cost; subsequent boots just memory-map.
    try:
        from retrieval.reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(model_id=settings.reranker_model)
        rr._load()  # eager-load; otherwise the model is lazy-instantiated on first rerank
        logger.info("Reranker pre-loaded", extra={"model": settings.reranker_model})
    except Exception as exc:
        logger.warning("Reranker pre-load failed (non-fatal)", extra={"err": str(exc)})

    yield

    logger.info("Shutting down Agentic RAG API")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.api_title,
    description=(
        "Production-grade Agentic RAG using LangGraph. "
        "Features: hybrid retrieval (BM25 + vector + RRF), cross-encoder reranking, "
        "self-grading, query rewriting, hallucination detection, confidence scoring, "
        "sentence-level citations, SSE streaming, RAGAS evaluation."
    ),
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

def _route_template(request: Request) -> str:
    """Return the matched route template, not the raw URL.

    Using ``request.url.path`` directly (e.g. ``/documents/foo.pdf``)
    creates a new Prometheus time-series for every distinct filename and
    explodes the cardinality. The matched route from the FastAPI router
    (``/documents/{filename}``) is bounded.
    """
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    return request.url.path


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    """Bind an X-Request-Id correlation ID and record per-request metrics.

    Correlation ID:
      - If the caller supplied X-Request-Id, we honour it (chained services).
      - Otherwise we generate a 12-char hex ID.
      - The ID is propagated to every log emitted during this request via
        the contextvar in core/logging.py — including from sync code run
        inside loop.run_in_executor.
      - We echo it in the response so the client can quote it in incident
        reports.

    Metrics:
      Prometheus counters + histograms via infrastructure.prometheus.
      Labels are bounded — route template (not raw path), method, status
      code, tenant_id (resolved best-effort from header).
    """
    incoming_id = request.headers.get("x-request-id")
    rid = set_request_id(incoming_id)
    api_key = request.headers.get("x-api-key")
    tenant_label = (
        get_settings().api_key_map.get(api_key, "unknown")
        if api_key else get_settings().default_tenant
    )

    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.time() - start) * 1000
        logger.exception("Unhandled request error", extra={
            "path": request.url.path,
            "method": request.method,
            "latency_ms": round(elapsed, 1),
        })
        raise

    elapsed_ms = (time.time() - start) * 1000

    path_template = _route_template(request)
    HTTP_REQUESTS.labels(
        method=request.method,
        path=path_template,
        status=str(response.status_code),
        tenant=tenant_label,
    ).inc()
    HTTP_LATENCY.labels(
        method=request.method,
        path=path_template,
        tenant=tenant_label,
    ).observe(elapsed_ms / 1000.0)

    if response.status_code == 401:
        AUTH_REJECTIONS.labels(reason="missing_or_invalid_key").inc()

    response.headers["x-request-id"] = rid

    logger.info("Request complete", extra={
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "latency_ms": round(elapsed_ms, 1),
    })
    return response


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus text-exposition endpoint.

    Excluded from the OpenAPI schema deliberately — scrapers don't read
    OpenAPI, and exposing it pollutes the customer-facing API docs.
    """
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)

app.include_router(health.router)
app.include_router(documents.router)   
app.include_router(query.router)
app.include_router(evaluation.router)
