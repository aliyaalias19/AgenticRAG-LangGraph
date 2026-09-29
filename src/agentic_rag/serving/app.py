"""FastAPI application with SSE streaming and tenant-scoped access."""

import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from agentic_rag.agent.graph import AgentRunner, total_duration_ms, trace_summary
from agentic_rag.agent.nodes import GraphDeps
from agentic_rag.agent.state import Termination
from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.security.audit import AUDIT_FILENAME, append_record, build_record
from agentic_rag.security.tenancy import (
    AuthenticationError,
    TenantContext,
    resolve_tenant,
)
from agentic_rag.serving import telemetry
from agentic_rag.serving.contracts import (
    Citation,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    StreamEventType,
)

logger = get_logger(__name__)

VERSION = "0.1.0"


@dataclass
class ServiceContext:
    """Long-lived objects shared by every request."""

    runner: AgentRunner
    settings: Settings

    @property
    def audit_path(self) -> Any:
        return self.settings.paths.data_dir / AUDIT_FILENAME


def create_app(
    runner_factory: Callable[[Settings], AgentRunner] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the application.

    The runner is injected rather than constructed here so that tests can
    drive the full HTTP surface with a scripted agent, and so that starting
    the process does not require a vector store to be reachable.
    """
    settings = settings or get_settings()
    configure_logging()

    app = FastAPI(
        title="Agentic RAG",
        version=VERSION,
        description="Tenant-scoped agentic retrieval over Kubernetes documentation",
    )

    def _default_factory(config: Settings) -> AgentRunner:
        from agentic_rag.llm.provider import build_provider
        from agentic_rag.retrieval.hybrid import build_retriever

        return AgentRunner(
            GraphDeps(
                provider=build_provider(config),
                retriever=build_retriever(config),
                settings=config,
            )
        )

    factory = runner_factory or _default_factory

    # Built eagerly rather than on a startup event. Every component that
    # touches the network -- the Qdrant client, the embedding model, the
    # reranker -- is lazily initialised behind a cached property, so
    # construction is cheap and importing the app never opens a socket.
    # Deferring to a startup hook only means the service is missing whenever
    # the app is exercised outside a full ASGI lifespan.
    app.state.service = ServiceContext(runner=factory(settings), settings=settings)
    logger.info("service_created", version=VERSION, provider=settings.llm.provider)

    def get_service(request: Request) -> ServiceContext:
        service: ServiceContext = request.app.state.service
        return service

    def get_tenant(
        x_api_key: str = Header(default="", alias="X-API-Key"),
        service: ServiceContext = Depends(get_service),
    ) -> TenantContext:
        """Resolve the caller's tenant from the API key header."""
        if not x_api_key:
            telemetry.AUTH_FAILURES.inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header",
            )
        try:
            return resolve_tenant(x_api_key, service.settings)
        except AuthenticationError:
            telemetry.AUTH_FAILURES.inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            ) from None

    def _citations(state: dict[str, Any]) -> list[Citation]:
        return [
            Citation(
                chunk_id=hit.chunk_id,
                title=hit.doc_title,
                heading_path=list(hit.heading_path),
                source_path=hit.source_path,
                language=hit.language,
            )
            for hit in state.get("context", [])
        ]

    def _record(
        service: ServiceContext,
        tenant: TenantContext,
        request_id: str,
        state: dict[str, Any],
    ) -> None:
        """Emit telemetry and an audit record for a completed run."""
        outcome = str(state.get("termination", Termination.RUNNING))
        label = tenant.tenant_id

        telemetry.REQUESTS.labels(tenant=label, outcome=outcome).inc()
        telemetry.RETRIEVED_CHUNKS.labels(tenant=label).observe(len(state.get("context", [])))
        for _ in range(int(state.get("rewrite_count", 0))):
            telemetry.REWRITES.labels(tenant=label).inc()
        if state.get("suppressed"):
            telemetry.SUPPRESSED.labels(tenant=label).inc()

        if service.settings.security.audit_log_enabled:
            append_record(
                build_record(
                    tenant,
                    action="query",
                    request_id=request_id,
                    outcome=outcome,
                    context_chunks=len(state.get("context", [])),
                    rewrites=int(state.get("rewrite_count", 0)),
                    suppressed=bool(state.get("suppressed", False)),
                ),
                service.audit_path,
            )

    @app.get("/healthz", response_model=HealthResponse)
    def healthz(service: ServiceContext = Depends(get_service)) -> HealthResponse:
        """Report liveness and the active configuration."""
        deps = service.runner.deps
        return HealthResponse(
            status="ok",
            version=VERSION,
            provider=deps.provider.name,
            collection=service.settings.vector_store.collection_name,
            lexical_index_loaded=getattr(deps.retriever, "lexical", None) is not None,
        )

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        """Expose Prometheus metrics."""
        return PlainTextResponse(telemetry.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        tenant: TenantContext = Depends(get_tenant),
        service: ServiceContext = Depends(get_service),
    ) -> QueryResponse:
        """Answer a question and return the complete result."""
        request_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id, **tenant.as_log_fields())
        started = time.perf_counter()

        try:
            state = service.runner.run(payload.question, tenant, request_id, payload.language)
            elapsed = time.perf_counter() - started
            telemetry.LATENCY.labels(tenant=tenant.tenant_id).observe(elapsed)
            _record(service, tenant, request_id, dict(state))

            return QueryResponse(
                request_id=request_id,
                answer=state.get("answer", ""),
                citations=_citations(dict(state)),
                query_type=str(state.get("query_type", "")),
                termination=str(state.get("termination", "")),
                verdict=str(state.get("verdict", "")),
                faithfulness=float(state.get("faithfulness", 0.0)),
                suppressed=bool(state.get("suppressed", False)),
                rewrite_count=int(state.get("rewrite_count", 0)),
                duration_ms=total_duration_ms(state),
                trace=trace_summary(state) if payload.include_trace else None,
            )
        finally:
            structlog.contextvars.clear_contextvars()

    @app.post("/v1/query/stream")
    def query_stream(
        payload: QueryRequest,
        tenant: TenantContext = Depends(get_tenant),
        service: ServiceContext = Depends(get_service),
    ) -> StreamingResponse:
        """Answer a question, streaming progress and tokens as they arrive.

        The graph itself is not incremental -- retrieval and grading must
        finish before an answer exists -- so status events carry the wait and
        the answer is chunked afterwards. A client that shows nothing until
        the first token would otherwise look frozen for several seconds.
        """
        request_id = str(uuid.uuid4())

        async def events() -> AsyncIterator[str]:
            started = time.perf_counter()
            structlog.contextvars.bind_contextvars(request_id=request_id, **tenant.as_log_fields())
            try:
                yield StreamEvent(
                    type=StreamEventType.STATUS,
                    data={"stage": "retrieving", "request_id": request_id},
                ).encode()

                state = service.runner.run(payload.question, tenant, request_id, payload.language)
                state_dict: dict[str, Any] = dict(state)

                yield StreamEvent(
                    type=StreamEventType.STATUS,
                    data={
                        "stage": "generating",
                        "context_chunks": len(state_dict.get("context", [])),
                        "rewrites": int(state_dict.get("rewrite_count", 0)),
                    },
                ).encode()

                telemetry.TIME_TO_FIRST_TOKEN.labels(tenant=tenant.tenant_id).observe(
                    time.perf_counter() - started
                )

                answer = str(state_dict.get("answer", ""))
                for start in range(0, len(answer), 48):
                    yield StreamEvent(
                        type=StreamEventType.TOKEN,
                        data={"text": answer[start : start + 48]},
                    ).encode()

                yield StreamEvent(
                    type=StreamEventType.CITATIONS,
                    data={"citations": [c.model_dump() for c in _citations(state_dict)]},
                ).encode()

                if payload.include_trace:
                    yield StreamEvent(
                        type=StreamEventType.TRACE,
                        data={"steps": trace_summary(state)},
                    ).encode()

                elapsed = time.perf_counter() - started
                telemetry.LATENCY.labels(tenant=tenant.tenant_id).observe(elapsed)
                _record(service, tenant, request_id, state_dict)

                yield StreamEvent(
                    type=StreamEventType.DONE,
                    data={
                        "request_id": request_id,
                        "termination": str(state_dict.get("termination", "")),
                        "suppressed": bool(state_dict.get("suppressed", False)),
                        "duration_ms": round(elapsed * 1000, 2),
                    },
                ).encode()

            except Exception as error:
                logger.exception("stream_failed", request_id=request_id)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"message": type(error).__name__},
                ).encode()
            finally:
                structlog.contextvars.clear_contextvars()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
