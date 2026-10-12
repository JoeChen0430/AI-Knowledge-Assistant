"""Model router: the degradation chain.

Tiers, tried in order:

  0  primary model                  full quality
  1..n  fallback models             smaller/faster, still generative
  --   (caller's job) stale cache   answered before, evidence unchanged
  --   (caller's job) extractive    return retrieved passages, no generation

Each model gets its own circuit breaker, so a wedged primary opens after a few
failures and subsequent requests skip straight to the fallback instead of
paying the timeout every time. The router never raises for a *degradation* —
it returns a result carrying ``tier`` and ``degrade_reason`` so the caller and
the UI can tell the user what they are looking at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from aka.llm.breaker import CircuitOpenError, get_breaker
from aka.llm.client import (
    ChatMessage,
    Completion,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    OllamaClient,
)
from aka.llm.faults import build_client
from aka.obs.logging import get_logger
from aka.obs.metrics import degradations, llm_calls, llm_latency, llm_tokens
from aka.obs.telemetry import (
    ATTR_DEGRADE_REASON,
    ATTR_DEGRADED,
    ATTR_MODEL,
    ATTR_MODEL_TIER,
    ATTR_TOKENS_IN,
    ATTR_TOKENS_OUT,
    set_attrs,
    stage,
)
from aka.settings import Settings, get_settings

log = get_logger("aka.llm.router")

# Failures that mean "the dependency is unhealthy" and should count toward
# opening the breaker. A plain LLMError (e.g. 400 bad request) does not.
_COUNTED = (LLMUnavailable, LLMTimeout)


class AllModelsUnavailable(RuntimeError):
    """Every generative tier failed. Caller must degrade non-generatively."""

    def __init__(self, attempts: list["AttemptRecord"]):
        self.attempts = attempts
        detail = "; ".join(f"{a.model}: {a.error}" for a in attempts) or "no models configured"
        super().__init__(f"all model tiers failed ({detail})")


@dataclass
class AttemptRecord:
    model: str
    tier: int
    error: str
    elapsed_s: float


@dataclass
class GenerationResult:
    text: str
    model: str
    tier: int
    degraded: bool
    degrade_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if not self.degraded:
            return "primary"
        return "fallback_model"


class ModelRouter:
    def __init__(self, settings: Settings | None = None, client: OllamaClient | None = None):
        self.settings = settings or get_settings()
        self.client = client or build_client(self.settings)
        self._chain: list[str] = [self.settings.LLM_MODEL, *self.settings.LLM_FALLBACK_MODELS]

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    def _breaker(self, model: str):
        return get_breaker(
            f"llm:{model}",
            failure_threshold=self.settings.BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=self.settings.BREAKER_RECOVERY_TIMEOUT_S,
            half_open_max_calls=self.settings.BREAKER_HALF_OPEN_MAX_CALLS,
            counted_exceptions=_COUNTED,
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Walk the chain until one model answers.

        ``deadline`` is an absolute ``time.monotonic()` value: the per-call
        timeout is clamped to the remaining budget so a slow primary cannot
        consume the whole request and leave nothing for the fallback.
        """
        attempts: list[AttemptRecord] = []

        for tier, model in enumerate(self._chain):
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 1.0:
                attempts.append(AttemptRecord(model, tier, "request budget exhausted", 0.0))
                break

            timeout = self._call_timeout(remaining, tier)
            started = time.perf_counter()

            with stage(
                "llm.generate",
                **{ATTR_MODEL: model, ATTR_MODEL_TIER: tier},
            ) as span:
                try:
                    completion = self._breaker(model).call(
                        self.client.chat,
                        list(messages),
                        model=model,
                        timeout=timeout,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except CircuitOpenError as exc:
                    elapsed = time.perf_counter() - started
                    llm_calls.labels(model=model, tier=str(tier), outcome="circuit_open").inc()
                    attempts.append(AttemptRecord(model, tier, "circuit open", elapsed))
                    log.warning("llm.circuit_open", model=model, retry_after=exc.retry_after)
                    continue
                except LLMError as exc:
                    elapsed = time.perf_counter() - started
                    outcome = "timeout" if isinstance(exc, LLMTimeout) else "error"
                    llm_calls.labels(model=model, tier=str(tier), outcome=outcome).inc()
                    llm_latency.labels(model=model).observe(elapsed)
                    attempts.append(AttemptRecord(model, tier, f"{type(exc).__name__}: {exc}", elapsed))
                    log.warning("llm.attempt_failed", model=model, tier=tier, error=str(exc))
                    continue

                elapsed = time.perf_counter() - started
                if not completion.text.strip():
                    # An empty body is a failure even though HTTP said 200.
                    llm_calls.labels(model=model, tier=str(tier), outcome="empty").inc()
                    attempts.append(AttemptRecord(model, tier, "empty completion", elapsed))
                    continue

                llm_calls.labels(model=model, tier=str(tier), outcome="success").inc()
                llm_latency.labels(model=model).observe(elapsed)
                llm_tokens.labels(model=model, direction="in").inc(completion.tokens_in)
                llm_tokens.labels(model=model, direction="out").inc(completion.tokens_out)

                degraded = tier > 0
                reason = f"primary '{self._chain[0]}' unavailable" if degraded else None
                set_attrs(
                    span,
                    **{
                        ATTR_TOKENS_IN: completion.tokens_in,
                        ATTR_TOKENS_OUT: completion.tokens_out,
                        ATTR_DEGRADED: degraded,
                        ATTR_DEGRADE_REASON: reason,
                    },
                )
                if degraded:
                    degradations.labels(reason="model_fallback", mode="fallback_model").inc()
                    log.info("llm.degraded", model=model, tier=tier, attempts=len(attempts))

                return GenerationResult(
                    text=completion.text,
                    model=model,
                    tier=tier,
                    degraded=degraded,
                    degrade_reason=reason,
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    latency_s=elapsed,
                    attempts=attempts,
                )

        raise AllModelsUnavailable(attempts)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        deadline: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> tuple[Iterator[str], str, int]:
        """Streaming variant. Returns (token_iterator, model, tier).

        Fallback only covers failures *before* the first token: once bytes have
        been sent to the client, silently switching models would splice two
        different answers together. After first token, a mid-stream failure
        surfaces to the caller.
        """
        attempts: list[AttemptRecord] = []

        for tier, model in enumerate(self._chain):
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 1.0:
                break
            timeout = self._call_timeout(remaining, tier)
            breaker = self._breaker(model)

            try:
                breaker._before_call()  # reserve a slot without wrapping the generator
            except CircuitOpenError:
                llm_calls.labels(model=model, tier=str(tier), outcome="circuit_open").inc()
                attempts.append(AttemptRecord(model, tier, "circuit open", 0.0))
                continue

            try:
                # Construction is inside the try on purpose: a generator-based
                # client defers the error to next(), but a client that connects
                # eagerly raises right here, and both must fall through.
                gen = self.client.chat_stream(
                    list(messages),
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                first = next(gen)
            except StopIteration:
                breaker.record_failure()
                attempts.append(AttemptRecord(model, tier, "empty stream", 0.0))
                continue
            except LLMError as exc:
                if isinstance(exc, _COUNTED):
                    breaker.record_failure()
                attempts.append(AttemptRecord(model, tier, str(exc), 0.0))
                log.warning("llm.stream_failed", model=model, tier=tier, error=str(exc))
                continue

            breaker.record_success()
            llm_calls.labels(model=model, tier=str(tier), outcome="success").inc()
            if tier > 0:
                degradations.labels(reason="model_fallback", mode="fallback_model").inc()

            def _chain_iter(first_token=first, generator=gen):
                yield first_token
                yield from generator

            return _chain_iter(), model, tier

        raise AllModelsUnavailable(attempts)

    def health(self) -> dict:
        reachable = self.client.ping()
        return {
            "backend_reachable": reachable,
            "chain": self._chain,
            "breakers": [self._breaker(m).snapshot() for m in self._chain],
        }

    # --- budget helpers ---------------------------------------------------
    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    def _call_timeout(self, remaining: float | None, tier: int) -> float:
        base = self.settings.LLM_TIMEOUT_S
        # Later tiers are meant to be fast; don't let one burn the whole budget.
        tier_cap = base if tier == 0 else max(10.0, base / 2)
        if remaining is None:
            return tier_cap
        # No floor above the remaining budget: a fixed minimum here would let
        # the last attempt run past the request deadline. generate() does not
        # start an attempt with <= 1s left, so this stays >= 0.5s.
        return max(0.5, min(tier_cap, remaining - 0.5))


_router: ModelRouter | None = None


def get_router(settings: Settings | None = None) -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter(settings)
    return _router


def reset_router() -> None:
    global _router
    _router = None
