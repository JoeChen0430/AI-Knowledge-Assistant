"""Circuit breaker.

Prevents a dead Ollama from turning every request into a timeout wait. After
``failure_threshold`` consecutive failures the breaker opens and calls fail
instantly; after ``recovery_timeout`` it admits a small number of probe calls
(half-open) and closes again only if they succeed.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from aka.obs.logging import get_logger
from aka.obs.metrics import breaker_state, breaker_transitions

log = get_logger("aka.breaker")
T = TypeVar("T")


class State(str, enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_STATE_VALUE = {State.CLOSED: 0, State.HALF_OPEN: 1, State.OPEN: 2}


class CircuitOpenError(RuntimeError):
    """Raised instead of attempting a call the breaker knows will fail."""

    def __init__(self, name: str, retry_after: float):
        super().__init__(f"circuit '{name}' is open; retry in {retry_after:.1f}s")
        self.name = name
        self.retry_after = retry_after


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 2
    # Only these exception types count as failures; a ValueError from bad input
    # is a caller bug, not a sign the dependency is down.
    counted_exceptions: tuple[type[BaseException], ...] = (Exception,)
    time_source: Callable[[], float] = time.monotonic

    _state: State = field(default=State.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        breaker_state.labels(name=self.name).set(0)

    # --- introspection ----------------------------------------------------
    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def snapshot(self) -> dict:
        with self._lock:
            self._maybe_half_open()
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "retry_after": self._retry_after(),
            }

    # --- internals --------------------------------------------------------
    def _retry_after(self) -> float:
        if self._state is not State.OPEN:
            return 0.0
        return max(0.0, self.recovery_timeout - (self.time_source() - self._opened_at))

    def _maybe_half_open(self) -> None:
        if self._state is State.OPEN and self._retry_after() <= 0:
            self._transition(State.HALF_OPEN)
            self._half_open_calls = 0

    def _transition(self, new: State) -> None:
        if new is self._state:
            return
        log.info("breaker.transition", breaker=self.name, from_state=self._state.value, to_state=new.value)
        self._state = new
        breaker_state.labels(name=self.name).set(_STATE_VALUE[new])
        breaker_transitions.labels(name=self.name, to_state=new.value).inc()

    # --- call path --------------------------------------------------------
    def _before_call(self) -> None:
        with self._lock:
            self._maybe_half_open()
            if self._state is State.OPEN:
                raise CircuitOpenError(self.name, self._retry_after())
            if self._state is State.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(self.name, self.recovery_timeout)
                self._half_open_calls += 1

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state is not State.CLOSED:
                self._transition(State.CLOSED)
            self._half_open_calls = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is State.HALF_OPEN:
                # A failed probe re-opens immediately; don't wait for threshold.
                self._opened_at = self.time_source()
                self._transition(State.OPEN)
            elif self._failures >= self.failure_threshold:
                self._opened_at = self.time_source()
                self._transition(State.OPEN)

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        self._before_call()
        try:
            result = fn(*args, **kwargs)
        except self.counted_exceptions:
            self.record_failure()
            raise
        except Exception:
            # Uncounted exception: neither success nor dependency failure.
            raise
        self.record_success()
        return result

    async def call_async(self, fn: Callable[..., T], *args, **kwargs) -> T:
        self._before_call()
        try:
            result = await fn(*args, **kwargs)
        except self.counted_exceptions:
            self.record_failure()
            raise
        except Exception:
            raise
        self.record_success()
        return result

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_calls = 0
            self._transition(State.CLOSED)


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Process-wide breaker registry, so all callers of one dependency share state.

    Note that kwargs only apply on first registration — the point of the shared
    registry is that every caller of a dependency observes the same breaker, so
    a later caller cannot quietly reconfigure it out from under the others.
    """
    with _registry_lock:
        if name not in _registry:
            _registry[name] = CircuitBreaker(name=name, **kwargs)
        return _registry[name]


def all_breakers() -> list[dict]:
    with _registry_lock:
        return [b.snapshot() for b in _registry.values()]


def reset_all_breakers() -> None:
    """Close every breaker, keeping their configuration. Operator action."""
    with _registry_lock:
        for b in _registry.values():
            b.reset()


def clear_breaker_registry() -> None:
    """Forget every breaker, configuration included.

    Test hook: because get_breaker ignores kwargs for an already-registered
    name, leaving instances behind makes test outcomes depend on ordering.
    """
    with _registry_lock:
        _registry.clear()
