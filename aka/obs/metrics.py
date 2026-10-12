"""Prometheus metrics.

Declared once at import time. Names follow ``aka_<subsystem>_<unit>`` so a
single Grafana dashboard can be built off a ``aka_*`` prefix query.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# Latency buckets tuned for local LLM inference: sub-second retrieval up to
# multi-minute generation on a laptop.
_LLM_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)
_FAST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)

# --- request / pipeline ---------------------------------------------------
stage_duration = Histogram(
    "aka_stage_duration_seconds",
    "Wall time of one pipeline stage",
    ["stage"],
    buckets=_FAST_BUCKETS,
    registry=REGISTRY,
)
chat_requests = Counter(
    "aka_chat_requests_total",
    "Chat requests by outcome",
    ["outcome", "degraded"],
    registry=REGISTRY,
)
chat_latency = Histogram(
    "aka_chat_latency_seconds",
    "End-to-end chat latency",
    buckets=_LLM_BUCKETS,
    registry=REGISTRY,
)

# --- retrieval ------------------------------------------------------------
retrieval_results = Counter(
    "aka_retrieval_total",
    "Retrieval attempts labelled by whether anything cleared the score floor",
    ["hit"],
    registry=REGISTRY,
)
retrieval_top_score = Histogram(
    "aka_retrieval_top_score",
    "Similarity of the best-scoring retrieved chunk",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# --- llm ------------------------------------------------------------------
llm_calls = Counter(
    "aka_llm_calls_total",
    "LLM calls by model, fallback tier and outcome",
    ["model", "tier", "outcome"],
    registry=REGISTRY,
)
llm_latency = Histogram(
    "aka_llm_latency_seconds",
    "LLM call latency",
    ["model"],
    buckets=_LLM_BUCKETS,
    registry=REGISTRY,
)
llm_tokens = Counter(
    "aka_llm_tokens_total",
    "Tokens consumed, by direction",
    ["model", "direction"],
    registry=REGISTRY,
)
answer_cache = Counter(
    "aka_answer_cache_total",
    "Answer cache lookups",
    ["result"],
    registry=REGISTRY,
)

# --- resilience -----------------------------------------------------------
degradations = Counter(
    "aka_degradations_total",
    "Times the pipeline served a degraded response",
    ["reason", "mode"],
    registry=REGISTRY,
)
breaker_state = Gauge(
    "aka_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["name"],
    registry=REGISTRY,
)
breaker_transitions = Counter(
    "aka_circuit_breaker_transitions_total",
    "Circuit breaker state transitions",
    ["name", "to_state"],
    registry=REGISTRY,
)
embed_requests = Counter(
    "aka_embed_requests_total",
    "Embedding requests by outcome (success, retry, cache_hit, unavailable)",
    ["outcome"],
    registry=REGISTRY,
)
fault_injections = Counter(
    "aka_fault_injections_total",
    "Faults injected by the chaos wrapper, by call target and fault kind",
    ["target", "kind"],
    registry=REGISTRY,
)

# --- ingestion ------------------------------------------------------------
ingest_jobs = Counter(
    "aka_ingest_jobs_total",
    "Ingest jobs by terminal status",
    ["status"],
    registry=REGISTRY,
)
ingest_duration = Histogram(
    "aka_ingest_duration_seconds",
    "Ingest job duration",
    buckets=_LLM_BUCKETS,
    registry=REGISTRY,
)
ingest_queue_depth = Gauge(
    "aka_ingest_queue_depth",
    "Jobs waiting in the ingest queue",
    ["state"],
    registry=REGISTRY,
)
ingest_chunks = Counter(
    "aka_ingest_chunks_total",
    "Chunks handled during ingest, by action (embedded vs skipped)",
    ["action"],
    registry=REGISTRY,
)
index_swaps = Counter(
    "aka_index_swaps_total",
    "Blue/green index swaps by outcome",
    ["outcome"],
    registry=REGISTRY,
)

# --- tiered memory --------------------------------------------------------
memory_writes = Counter(
    "aka_memory_writes_total",
    "L2 memory write attempts by outcome",
    ["outcome"],
    registry=REGISTRY,
)
memory_items = Gauge(
    "aka_memory_items",
    "Live L2 memory items",
    ["tenant"],
    registry=REGISTRY,
)
memory_promotions = Counter(
    "aka_memory_promotions_total",
    "L1 to L2 promotions by decision",
    ["decision"],
    registry=REGISTRY,
)

# --- human-in-the-loop ----------------------------------------------------
hitl_decisions = Counter(
    "aka_hitl_decisions_total",
    "Policy engine verdicts",
    ["rule", "decision"],
    registry=REGISTRY,
)
hitl_queue_depth = Gauge(
    "aka_hitl_queue_depth",
    "Pending approvals",
    ["kind"],
    registry=REGISTRY,
)
hitl_overrides = Counter(
    "aka_hitl_overrides_total",
    "Human verdicts on queued items — the override-rate numerator",
    ["kind", "verdict"],
    registry=REGISTRY,
)
hitl_resolution_seconds = Histogram(
    "aka_hitl_resolution_seconds",
    "Time an approval sat in the queue before a human resolved it",
    buckets=(1, 10, 60, 300, 1800, 3600, 21600, 86400),
    registry=REGISTRY,
)

# --- security -------------------------------------------------------------
auth_failures = Counter(
    "aka_auth_failures_total",
    "Rejected requests by reason",
    ["reason"],
    registry=REGISTRY,
)
rate_limited = Counter(
    "aka_rate_limited_total",
    "Requests rejected by the rate limiter",
    ["scope"],
    registry=REGISTRY,
)
injection_findings = Counter(
    "aka_injection_findings_total",
    "Prompt-injection heuristics that fired",
    ["surface", "rule"],
    registry=REGISTRY,
)
scan_findings = Counter(
    "aka_scan_findings_total",
    "Secret/PII detections at ingest",
    ["kind"],
    registry=REGISTRY,
)


@contextmanager
def timed(stage_name: str) -> Iterator[None]:
    """Record stage wall time even when the block raises."""
    start = time.perf_counter()
    try:
        yield
    finally:
        stage_duration.labels(stage=stage_name).observe(time.perf_counter() - start)


def render() -> tuple[bytes, str]:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
