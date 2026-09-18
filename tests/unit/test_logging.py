"""Tests for structured logging configuration."""

import structlog

from agentic_rag.obs.logging import configure_logging, get_logger


def test_get_logger_returns_bound_logger() -> None:
    logger = get_logger("test")
    assert logger is not None


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    first = structlog.get_config()["processors"]
    configure_logging()
    second = structlog.get_config()["processors"]
    assert len(first) == len(second)


def test_log_capture_records_event_and_fields() -> None:
    configure_logging(force=True)
    with structlog.testing.capture_logs() as logs:
        get_logger("test").info("corpus_ingested", document_count=42)

    assert len(logs) == 1
    assert logs[0]["event"] == "corpus_ingested"
    assert logs[0]["document_count"] == 42


def test_merge_contextvars_processor_is_configured() -> None:
    configure_logging(force=True)
    processors = structlog.get_config()["processors"]
    assert structlog.contextvars.merge_contextvars in processors


def test_contextvars_are_bound_and_cleared() -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(tenant_id="tenant-a")
    assert structlog.contextvars.get_contextvars()["tenant_id"] == "tenant-a"

    structlog.contextvars.clear_contextvars()
    assert "tenant_id" not in structlog.contextvars.get_contextvars()
