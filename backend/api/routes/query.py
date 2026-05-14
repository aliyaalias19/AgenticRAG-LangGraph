"""
Query routes: standard and streaming RAG queries, plus the supervisor
intent classifier. Multi-agent orchestration was deliberately removed —
see the placeholder block below the streaming route for the rationale.
"""

import json
import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from agents.graph import run_rag, run_rag_streaming
from core.audit import audit_event
from core.auth import TenantContext, get_tenant_context
from core.config import get_settings
from core.exceptions import GradingTimeout, RAGError, RetrievalError
from ingestion.pipeline import list_ingested_files
from models.domain import RetrievalMetadata
from models.requests import QueryRequest, SupervisorRequest
from models.responses import CitationResponse, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)


def _require_documents(tenant_id: str) -> None:
    if not list_ingested_files(settings, tenant_id=tenant_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload a PDF via POST /documents first.",
        )


# ─────────────────────────────────────────
# Supervisor — keyword-only intent classification
# ─────────────────────────────────────────
# Earlier versions classified intent via an LLM round-trip per chat message
# with a keyword fallback. Now that the orchestrate endpoint is gone and
# every intent routes through the same /query pipeline, the LLM call was
# pure overhead — it paid tokens to set a UI badge. Keyword-only is
# deterministic, free, and ~5ms. The endpoint stays so the frontend keeps
# its 'intent' chip without round-tripping; clients that need richer NLU
# can call /query directly with a more specific question.


def _classify_intent(msg: str) -> dict:
    m = msg.lower()
    if any(w in m for w in ["list", "what documents", "what files"]):
        return {"intent": "list_docs", "reasoning": "keyword: list/documents"}
    if any(w in m for w in ["summarise", "summarize", "summary", "tldr"]):
        return {"intent": "summarise", "reasoning": "keyword: summarise"}
    if any(w in m for w in ["risk", "analyse", "analyze", "recommend"]):
        return {"intent": "analyse", "reasoning": "keyword: analyse"}
    if any(w in m for w in ["hello", "hi", "hey", "thank"]):
        return {"intent": "chitchat", "reasoning": "keyword: greeting"}
    return {"intent": "question", "reasoning": "default"}


@router.post("/supervisor/classify", summary="Classify message intent (keyword-only)")
async def supervisor_classify(request: SupervisorRequest):
    return _classify_intent(request.message)


# ─────────────────────────────────────────
# Standard query (non-streaming)
# ─────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Agentic RAG query")
@limiter.limit("20/minute")
async def query(
    request: Request,
    body: QueryRequest,
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Run the full LangGraph agentic RAG pipeline and return structured results.

    Retrieval is scoped to the caller's tenant — answers never include
    citations from another tenant's corpus.

    The pipeline: analyze_query → retrieve → grade → [rewrite|generate] → hallucination_check

    Quality signals returned:
      relevance_score:    1-10, how well retrieved chunks matched the query
      confidence_score:   0-1,  composite of retrieval quality + grounding
      hallucination_score: 0-1, estimated fraction of unsupported claims
    """
    _require_documents(tenant.tenant_id)

    import asyncio
    loop = asyncio.get_running_loop()

    try:
        state = await loop.run_in_executor(
            None, partial(run_rag, body.question, None, tenant.tenant_id),
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}")
    except GradingTimeout as exc:
        raise HTTPException(status_code=504, detail=f"Grading timed out: {exc}")
    except RAGError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error for query: %r", body.question)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    meta_raw = state.get("retrieval_metadata") or {}
    retrieval_meta = None
    if meta_raw and "error" not in meta_raw:
        try:
            retrieval_meta = RetrievalMetadata(**{
                "query_used": state.get("rewritten_query", body.question),
                "vector_results_count": meta_raw.get("vector_results_count", 0),
                "bm25_results_count": meta_raw.get("bm25_results_count", 0),
                "fused_results_count": meta_raw.get("fused_results_count", 0),
                "reranking_applied": meta_raw.get("reranking_applied", False),
                "retrieval_latency_ms": meta_raw.get("retrieval_latency_ms", 0.0),
            })
        except Exception:
            pass

    citations = [
        CitationResponse(**c) if isinstance(c, dict) else CitationResponse(**c.model_dump())
        for c in state.get("citations", [])
    ]

    audit_event(
        "query",
        tenant_id=tenant.tenant_id,
        actor_fingerprint=tenant.api_key_fingerprint,
        detail={
            "question": body.question[:500],
            "citations": len(citations),
            "relevance_score": state.get("relevance_score", 0),
            "hallucination_score": state.get("hallucination_score", 0.0),
        },
    )

    return QueryResponse(
        answer=state.get("answer", ""),
        citations=citations,
        original_query=body.question,
        rewritten_query=state.get("rewritten_query", body.question),
        relevance_score=state.get("relevance_score", 0),
        hallucination_score=state.get("hallucination_score", 0.0),
        steps_taken=state.get("steps_taken", 0),
        reasoning_trace=state.get("reasoning_trace", []) if body.show_reasoning else [],
        latency_ms=state.get("latency_ms", 0.0),
        retrieval_metadata=retrieval_meta,
    )


# ─────────────────────────────────────────
# Streaming query (SSE)
# ─────────────────────────────────────────

@router.post("/query/stream", summary="Streaming RAG query (SSE)")
@limiter.limit("10/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Stream the RAG answer token-by-token via Server-Sent Events.

    Tenant-scoped: retrieval honours the caller's tenant_id.

    Event types (newline-delimited JSON):
      trace:     {"type":"trace","steps":[...]}
      token:     {"type":"token","content":"..."}
      citations: {"type":"citations","data":[...]}
      meta:      {"type":"meta","relevance_score":N,...}
      error:     {"type":"error","message":"..."}
    """
    _require_documents(tenant.tenant_id)
    audit_event(
        "query_stream",
        tenant_id=tenant.tenant_id,
        actor_fingerprint=tenant.api_key_fingerprint,
        detail={"question": body.question[:500]},
    )

    def event_stream():
        try:
            for line in run_rag_streaming(body.question, tenant_id=tenant.tenant_id):
                yield f"data: {line}\n"
        except Exception as exc:
            logger.exception("Streaming error for: %r", body.question)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# /agent/orchestrate was removed deliberately. It wrapped RAG with a
# single templated LLM pass and was framed as "multi-agent orchestration"
# in the original code, which it was not. Summarisation and analysis
# styles are now achieved by writing the right question to /query — the
# graph already does the work, and we don't ship endpoints that lie
# about their architecture. See README for the rationale.
