"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from agentic_rag.config.settings import get_settings

_configured = False


def configure_logging(*, force: bool = False) -> None:
    """Configure structlog and the stdlib logging bridge.

    Idempotent: repeated calls are no-ops unless ``force`` is set.
    """
    global _configured
    if _configured and not force:
        return

    settings = get_settings()

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.logging.json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.logging.level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.logging.level,
    )

    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)
