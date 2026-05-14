"""
Structured JSON logging with correlation IDs.

Why structured logs matter for this system
──────────────────────────────────────────
Every user query touches six pipeline stages (analyze → retrieve → grade →
[rewrite] → generate → hallucination_check). When a customer reports
"yesterday at 3pm I got the wrong answer," operations needs to pull every
log line emitted by every stage for that single request.

Plain ``logging.info("RAG complete: ...")`` makes that impossible — there
is no shared key correlating the stages, and the unstructured message
text resists machine-readable filtering.

This module installs:

  1. A ``JsonFormatter`` that emits one well-formed JSON object per line,
     with a stable schema: ``time, level, logger, msg, request_id``, plus
     any ``extra={...}`` fields the caller attaches.

  2. A ``request_id`` contextvar that is populated by the FastAPI
     middleware on every incoming request and propagated automatically to
     every log emitted during that request — including from background
     thread-pool executors used by ``run_in_executor`` (contextvars copy
     across thread-pool boundaries).

  3. A ``RequestIdFilter`` that injects the current ``request_id`` into
     every ``LogRecord`` so the formatter can include it unconditionally.

Operational use
───────────────
Logs are stdout-only. In a real deployment the operator pipes them into
their log aggregator (Loki, ELK, Splunk). ``request_id`` becomes the
filter key for tracing a single request end-to-end. Adding OpenTelemetry
spans on top of this is a one-import change — kept out of scope for now.

Why not adopt OTel directly?
────────────────────────────
OTel is the correct long-term answer. Skipping it here keeps the
dependency surface small for the prototype while still solving the
correlation problem cleanly. Migration path: replace ``setup_logging``
with an OTel handler; ``request_id`` becomes the trace ID.
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Optional

# ── Correlation ID propagation ───────────────────────────────────────────────

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the current request's correlation ID, or '-' if none bound."""
    return _request_id.get()


def set_request_id(value: Optional[str] = None) -> str:
    """Bind a correlation ID for the current request context.

    If ``value`` is None or empty, a fresh 12-char hex ID is generated.
    Returns the bound ID so callers can echo it in response headers.

    Thread-safety: contextvars are per-task and propagate to thread-pool
    executors via ``loop.run_in_executor`` automatically — no manual
    threading.local plumbing needed.
    """
    rid = value or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


# ── Formatter ────────────────────────────────────────────────────────────────

# These keys come from the stdlib logging.LogRecord and should not be echoed
# as user fields — they describe the log infrastructure itself.
_RESERVED_RECORD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "getMessage", "message", "asctime",
    "taskName",
})


class JsonFormatter(logging.Formatter):
    """One well-formed JSON object per log line.

    Stable top-level schema:
        time:       ISO-8601 UTC, millisecond precision
        level:      INFO | DEBUG | ...
        logger:     dotted logger name
        msg:        rendered log message
        request_id: correlation ID, or "-" outside a request

    Any keyword passed via ``extra={...}`` is merged at the top level
    (after a JSON-serialisability check). Non-serialisable values fall
    back to ``str()`` rather than crashing the log call.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key in payload or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


class RequestIdFilter(logging.Filter):
    """Inject the current request_id contextvar onto every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


# ── Setup ────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Install JSON logging on the root logger.

    Idempotent — safe to call multiple times. Subsequent calls replace
    handlers, so tests and lifespan handlers can re-initialise cleanly.

    All third-party loggers (uvicorn, langchain, httpx) inherit this
    handler automatically via the root. Uvicorn's own access logger is
    silenced — its plaintext output would defeat the JSON pipeline.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Tame the noisy access logger; we already log via metrics_middleware.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
