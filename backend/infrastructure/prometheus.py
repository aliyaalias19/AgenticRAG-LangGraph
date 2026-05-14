"""
Prometheus instrumentation.

Why this exists
───────────────
The in-memory ``MetricsCollector`` in ``infrastructure/metrics.py`` is
useful for the ``GET /metrics`` JSON endpoint that ships with the
prototype, but it has two operational problems:

  1. It dies with the process. A pod restart wipes the history.
  2. It cannot be scraped by Prometheus, the de-facto observability
     standard in any serious enterprise deployment.

This module exposes the same signals via ``prometheus_client``, which
serialises to Prometheus' text exposition format. The recommended
deployment topology is one scraper (Prometheus / VictoriaMetrics /
Mimir) per cluster, scraping ``/metrics`` every 15-30 seconds.

Multi-process deployment
────────────────────────
``prometheus_client`` uses an in-process registry by default. That is
correct for the single-worker development setup. In production with
``uvicorn --workers N``, each worker has its own counters and a
naive scrape returns inconsistent samples.

The fix is the library's multiprocess mode. To enable it:

  1. Set ``PROMETHEUS_MULTIPROC_DIR`` to a writable directory shared
     across worker processes. In Docker this is usually a tmpfs mount
     at ``/tmp/prom`` declared in compose:

        environment:
          - PROMETHEUS_MULTIPROC_DIR=/tmp/prom
        tmpfs:
          - /tmp/prom

  2. Replace ``REGISTRY`` below with::

        from prometheus_client import multiprocess, CollectorRegistry
        REGISTRY = CollectorRegistry()
        multiprocess.MultiProcessCollector(REGISTRY)

  3. Ensure each worker calls ``multiprocess.mark_process_dead(pid)``
     on shutdown (uvicorn's lifespan handler is the right place).

We do not bake this in by default because (a) the directory path
depends on the operator's container layout and (b) it adds I/O on
every counter increment, which is wasted overhead for the single-
worker case. The single-line registry above is the right default;
the multi-process upgrade is documented here so the upgrade is
mechanical when the deployment shape changes.

Labels
──────
We label conservatively: ``method``, ``path_template``, ``status``,
``tenant_id``. Adding labels with unbounded cardinality (raw URL, full
query, user_id) would explode the time-series count and DoS the scraper
— this is the most common rookie mistake with Prometheus.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# Default buckets tuned for an interactive RAG endpoint:
#   <100ms       hot cache, cached embedding, no LLM
#   100-500ms    typical retrieval-only path
#   500ms-3s     small-context LLM generation
#   3-15s        large-context or remote-LLM path
#   >15s         pathological — investigate
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "Count of HTTP requests served, labelled by method/path/status/tenant.",
    labelnames=("method", "path", "status", "tenant"),
    registry=REGISTRY,
)

HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "path", "tenant"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Hybrid retrieval (BM25 + vector + RRF + rerank) latency in seconds.",
    labelnames=("tenant",),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

AUDIT_EVENTS = Counter(
    "rag_audit_events_total",
    "Audit log writes, labelled by action and tenant.",
    labelnames=("action", "tenant"),
    registry=REGISTRY,
)

AUTH_REJECTIONS = Counter(
    "rag_auth_rejections_total",
    "Requests rejected at the auth layer.",
    labelnames=("reason",),  # missing_key | invalid_key
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics route."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
