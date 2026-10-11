"""Structured logging with automatic trace correlation.

Every log line carries ``trace_id``/``span_id`` when emitted inside a span, so a
line in Loki links straight to the trace in Tempo.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

import structlog

from aka.obs.telemetry import current_span_id, current_trace_id
from aka.settings import Settings, get_settings

_lock = threading.Lock()
_configured = False

# Keys whose values must never reach a log sink.
_REDACT_KEYS = {
    "password", "secret", "token", "api_key", "authorization",
    "jwt", "cookie", "set-cookie", "jwt_secret",
}
_REDACTED = "***redacted***"


def _add_trace_context(_logger, _method, event_dict: dict[str, Any]) -> dict[str, Any]:
    tid = current_trace_id()
    if tid:
        event_dict["trace_id"] = tid
        event_dict["span_id"] = current_span_id()
    return event_dict


def _redact(_logger, _method, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def setup_logging(settings: Settings | None = None) -> None:
    global _configured
    settings = settings or get_settings()
    with _lock:
        if _configured:
            return

        renderer = (
            structlog.processors.JSONRenderer()
            if settings.LOG_JSON
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                _add_trace_context,
                _redact,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
            ),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )

        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        )
        # Third-party loggers are noisy at INFO and drown our own lines.
        for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "watchdog"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True


def get_logger(name: str = "aka") -> structlog.stdlib.BoundLogger:
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)


def bind_request_context(**kwargs: Any) -> None:
    """Bind values onto every log line for the rest of this request/task."""
    structlog.contextvars.bind_contextvars(**{k: v for k, v in kwargs.items() if v is not None})


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
