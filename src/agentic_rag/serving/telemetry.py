"""Prometheus metrics, labelled by tenant.

Every metric carries the tenant label so that per-tenant load, latency and
suppression rates are visible without correlating logs. The tenant label comes
from the authenticated context, so a caller cannot inflate another tenant's
series by manipulating a request field.
"""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "agentic_rag_requests_total",
    "Queries received",
    ["tenant", "outcome"],
    registry=REGISTRY,
)

LATENCY = Histogram(
    "agentic_rag_request_duration_seconds",
    "End-to-end query latency",
    ["tenant"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    registry=REGISTRY,
)

TIME_TO_FIRST_TOKEN = Histogram(
    "agentic_rag_ttft_seconds",
    "Time from request to first streamed token",
    ["tenant"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    registry=REGISTRY,
)

RETRIEVED_CHUNKS = Histogram(
    "agentic_rag_context_chunks",
    "Chunks placed in the generation context",
    ["tenant"],
    buckets=(0, 1, 2, 4, 6, 8, 12),
    registry=REGISTRY,
)

REWRITES = Counter(
    "agentic_rag_query_rewrites_total",
    "Query reformulations triggered by failed relevance checks",
    ["tenant"],
    registry=REGISTRY,
)

SUPPRESSED = Counter(
    "agentic_rag_suppressed_answers_total",
    "Answers withheld by post-generation verification",
    ["tenant"],
    registry=REGISTRY,
)

AUTH_FAILURES = Counter(
    "agentic_rag_auth_failures_total",
    "Requests rejected during authentication",
    registry=REGISTRY,
)


def render() -> bytes:
    """Return the metrics exposition payload."""
    return generate_latest(REGISTRY)
