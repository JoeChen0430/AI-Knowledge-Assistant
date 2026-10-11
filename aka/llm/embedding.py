"""Resilient embedding calls.

Embedding is the one model call every request depends on: without a query
vector there is no retrieval, and without retrieval none of the generation
fallbacks have anything to stand on. So it gets its own protection rather than
riding on the LLM router:

* a circuit breaker shared by every caller of the embedding model, so a dead
  backend fails fast instead of costing a timeout per request;
* a small retry budget with jittered exponential backoff for transient faults,
  clamped to the request deadline;
* an LRU of query vectors, so a repeated question needs no embedding call.

When all of that fails the caller gets :class:`EmbeddingUnavailable` and decides
how to degrade (the RAG pipeline falls back to keyword retrieval).
"""

from __future__ import annotations

import random
import threading
import time
from collections import OrderedDict
from typing import Any, Sequence

from aka.ingest.hashing import normalize_text
from aka.llm.breaker import CircuitOpenError, get_breaker
from aka.llm.client import LLMTimeout, LLMUnavailable
from aka.obs.logging import get_logger
from aka.obs.metrics import embed_requests
from aka.settings import Settings, get_settings

log = get_logger("aka.llm.embedding")

_TRANSIENT = (LLMUnavailable, LLMTimeout)


class EmbeddingUnavailable(LLMUnavailable):
    """Every attempt failed, or the embedding breaker is open."""


class _VectorLRU:
    def __init__(self, capacity: int):
        self.capacity = max(0, capacity)
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            vec = self._data.get(key)
            if vec is not None:
                self._data.move_to_end(key)
            return vec

    def put(self, key: str, vec: list[float]) -> None:
        if self.capacity == 0:
            return
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ResilientEmbedder:
    def __init__(self, client: Any, settings: Settings | None = None):
        self.client = client
        self.settings = settings or get_settings()
        self._queries = _VectorLRU(self.settings.QUERY_EMBED_CACHE_SIZE)

    @property
    def model(self) -> str:
        return self.settings.EMBED_MODEL

    def _breaker(self):
        return get_breaker(
            f"embed:{self.model}",
            failure_threshold=self.settings.BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=self.settings.BREAKER_RECOVERY_TIMEOUT_S,
            half_open_max_calls=self.settings.BREAKER_HALF_OPEN_MAX_CALLS,
            counted_exceptions=_TRANSIENT,
        )

    def embed(self, texts: Sequence[str], *, deadline: float | None = None) -> list[list[float]]:
        """Embed a batch, retrying transient failures within the budget.

        Non-transient errors (bad input, a bug in a fake) propagate untouched:
        retrying them would only repeat the failure more slowly.
        """
        if not texts:
            return []
        attempts = max(1, self.settings.EMBED_RETRY_ATTEMPTS)
        last_error: Exception | None = None

        for attempt in range(attempts):
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                vectors = self._breaker().call(self.client.embed, list(texts))
                embed_requests.labels(outcome="success").inc()
                return vectors
            except CircuitOpenError as exc:
                # Retrying against an open circuit would just burn the budget.
                embed_requests.labels(outcome="unavailable").inc()
                raise EmbeddingUnavailable(str(exc)) from exc
            except _TRANSIENT as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                embed_requests.labels(outcome="retry").inc()
                backoff = self.settings.EMBED_RETRY_BACKOFF_S * (2**attempt)
                backoff *= 0.5 + random.random()  # jitter: no synchronized retry storms
                if deadline is not None:
                    backoff = min(backoff, max(0.0, deadline - time.monotonic()))
                log.debug("embed.retry", attempt=attempt + 1, backoff_s=round(backoff, 3), error=str(exc))
                time.sleep(backoff)

        embed_requests.labels(outcome="unavailable").inc()
        raise EmbeddingUnavailable(
            f"embedding failed after {attempts} attempt(s): {last_error or 'request budget exhausted'}"
        ) from last_error

    def embed_query(self, text: str, *, deadline: float | None = None) -> list[float]:
        key = f"{self.model}\x00{normalize_text(text).casefold()}"
        cached = self._queries.get(key)
        if cached is not None:
            embed_requests.labels(outcome="cache_hit").inc()
            return cached
        vectors = self.embed([text], deadline=deadline)
        if not vectors:
            raise EmbeddingUnavailable("embedding backend returned no vector")
        self._queries.put(key, vectors[0])
        return vectors[0]
