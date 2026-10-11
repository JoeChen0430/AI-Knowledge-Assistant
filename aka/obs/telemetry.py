"""OpenTelemetry wiring.

One tracer provider for the whole process. Exports over OTLP/HTTP when a
collector endpoint is configured, otherwise keeps spans in-process so that
``trace_id`` correlation still works on a laptop with no Docker running.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from aka.settings import Settings, get_settings

_lock = threading.Lock()
_configured = False

# Span attribute names kept in one place so dashboards and tests agree.
ATTR_TENANT = "aka.tenant_id"
ATTR_USER = "aka.user_id"
ATTR_CONVERSATION = "aka.conversation_id"
ATTR_STAGE = "aka.stage"
ATTR_MODEL = "aka.llm.model"
ATTR_MODEL_TIER = "aka.llm.tier"
ATTR_TOKENS_IN = "aka.llm.tokens_in"
ATTR_TOKENS_OUT = "aka.llm.tokens_out"
ATTR_DEGRADED = "aka.degraded"
ATTR_DEGRADE_REASON = "aka.degrade_reason"
ATTR_CHUNK_IDS = "aka.retrieval.chunk_ids"
ATTR_SCORES = "aka.retrieval.scores"
ATTR_TOP_SCORE = "aka.retrieval.top_score"
ATTR_GROUNDING = "aka.answer.grounding"
ATTR_HITL_DECISION = "aka.hitl.decision"
ATTR_HITL_RULE = "aka.hitl.rule"


def setup_telemetry(settings: Settings | None = None) -> None:
    """Idempotently install the global tracer provider."""
    global _configured
    settings = settings or get_settings()
    with _lock:
        if _configured or not settings.OTEL_ENABLED:
            return

        resource = Resource.create(
            {
                "service.name": settings.SERVICE_NAME,
                "service.version": settings.SERVICE_VERSION,
                "deployment.environment": settings.ENV,
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(settings.OTEL_SAMPLE_RATIO)),
        )

        if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=f"{settings.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/traces"
                    )
                )
            )
        if settings.OTEL_CONSOLE_EXPORT:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _configured = True


def get_tracer(name: str = "aka") -> trace.Tracer:
    return trace.get_tracer(name)


def current_trace_id() -> str | None:
    """32-hex trace id of the active span, or None when unsampled/absent."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.span_id, "016x")


@contextmanager
def stage(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    **attributes: Any,
) -> Iterator[Span]:
    """Span for one pipeline stage, tagged so per-stage latency is queryable.

    Records exceptions and marks the span ERROR, then re-raises — observability
    must never change control flow.
    """
    tracer = get_tracer()
    attrs = {ATTR_STAGE: name, **{k: v for k, v in attributes.items() if v is not None}}
    with tracer.start_as_current_span(name, kind=kind, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_attrs(span: Span | None = None, **attributes: Any) -> None:
    """Attach attributes to a span, skipping Nones and empty collections."""
    target = span or trace.get_current_span()
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = [v for v in value if v is not None]
        target.set_attribute(key, value)


def _reset_for_tests() -> None:
    global _configured
    _configured = False
