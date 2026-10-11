"""Fault injection for the model backend.

Resilience claims are only worth something if they are measured against
failures the system did not choose. :class:`FaultInjectingClient` wraps the
real Ollama client and fails a configurable fraction of calls *before* they
reach the backend, with the same exception types the real client raises, so
the breaker, the retry budget and the degradation ladder all see a failure
they cannot tell apart from a genuine outage.

Faults are independent per call and drawn from a seeded RNG, so a run is
reproducible: the same seed and the same request sequence fail the same calls.

Enabled through settings (``FAULT_INJECTION_RATE`` > 0), and refused outright
in production by :func:`aka.settings.get_settings`.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter
from typing import Any, Iterable, Iterator

from aka.llm.client import LLMTimeout, LLMUnavailable, OllamaClient
from aka.obs.logging import get_logger
from aka.obs.metrics import fault_injections
from aka.settings import Settings, get_settings

log = get_logger("aka.llm.faults")

TARGETS = frozenset({"chat", "embed"})


class FaultInjectingClient:
    """Drop-in wrapper for :class:`OllamaClient` that fails a share of calls.

    Two fault kinds, chosen with equal probability:

    * ``unavailable`` — raised immediately, like a refused connection or a 5xx.
    * ``timeout`` — sleeps ``timeout_delay_s`` (capped at the call's own
      timeout) before raising, so request time budgets are actually spent.
    """

    def __init__(
        self,
        inner: Any,
        *,
        rate: float,
        seed: int | None = None,
        targets: Iterable[str] = TARGETS,
        timeout_delay_s: float = 0.5,
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"fault rate must be within [0, 1], got {rate}")
        unknown = set(targets) - TARGETS
        if unknown:
            raise ValueError(f"unknown fault targets: {sorted(unknown)}")
        self.inner = inner
        self.rate = rate
        self.targets = frozenset(targets)
        self.timeout_delay_s = timeout_delay_s
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.stats: Counter[str] = Counter()

    def __getattr__(self, name: str) -> Any:
        # Everything that is not a model call (settings, ping, close, ...)
        # passes straight through to the real client.
        return getattr(self.inner, name)

    # --- fault decision ---------------------------------------------------
    def _maybe_fail(self, target: str, timeout: float | None = None) -> None:
        if target not in self.targets or self.rate <= 0.0:
            return
        with self._lock:
            roll = self._rng.random()
            kind_roll = self._rng.random()
            self.stats[f"{target}.calls"] += 1
            if roll >= self.rate:
                return
            kind = "timeout" if kind_roll < 0.5 else "unavailable"
            self.stats[f"{target}.faults"] += 1
            self.stats[f"{target}.{kind}"] += 1

        fault_injections.labels(target=target, kind=kind).inc()
        if kind == "timeout":
            delay = self.timeout_delay_s if timeout is None else min(self.timeout_delay_s, timeout)
            time.sleep(max(0.0, delay))
            raise LLMTimeout(f"[injected] {target} timed out")
        raise LLMUnavailable(f"[injected] {target} backend unavailable")

    def fault_summary(self) -> dict[str, int]:
        with self._lock:
            return dict(self.stats)

    # --- wrapped calls ----------------------------------------------------
    def chat(self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2):
        self._maybe_fail("chat", timeout)
        return self.inner.chat(
            messages, model=model, timeout=timeout, max_tokens=max_tokens, temperature=temperature
        )

    def chat_stream(
        self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2
    ) -> Iterator[str]:
        # Raised eagerly, before the first token, which is the case the
        # router's streaming fallback is built to handle.
        self._maybe_fail("chat", timeout)
        return self.inner.chat_stream(
            messages, model=model, timeout=timeout, max_tokens=max_tokens, temperature=temperature
        )

    def embed(self, texts, *, model=None):
        self._maybe_fail("embed")
        return self.inner.embed(texts, model=model)

    def embed_one(self, text, *, model=None):
        # Routed through self.embed so the single-text path is faulted too.
        vectors = self.embed([text], model=model)
        if not vectors:
            raise LLMUnavailable("no embedding returned")
        return vectors[0]


def wrap_with_faults(client: Any, settings: Settings) -> Any:
    """Wrap ``client`` when fault injection is configured, else return it."""
    if settings.FAULT_INJECTION_RATE <= 0.0:
        return client
    log.warning(
        "faults.enabled",
        rate=settings.FAULT_INJECTION_RATE,
        seed=settings.FAULT_INJECTION_SEED,
        targets=sorted(settings.FAULT_INJECTION_TARGETS),
    )
    return FaultInjectingClient(
        client,
        rate=settings.FAULT_INJECTION_RATE,
        seed=settings.FAULT_INJECTION_SEED,
        targets=settings.FAULT_INJECTION_TARGETS,
        timeout_delay_s=settings.FAULT_INJECTION_TIMEOUT_DELAY_S,
    )


def build_client(settings: Settings | None = None) -> Any:
    """The model client every component should use: real Ollama, plus the
    chaos wrapper when ``FAULT_INJECTION_RATE`` is set."""
    settings = settings or get_settings()
    return wrap_with_faults(OllamaClient(settings), settings)
