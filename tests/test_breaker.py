"""Circuit breaker state machine tests (deterministic clock, no sleeps)."""

from __future__ import annotations

import pytest

from aka.llm.breaker import CircuitBreaker, CircuitOpenError, State


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Boom(RuntimeError):
    pass


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def breaker(clock):
    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max_calls=2,
        time_source=clock,
    )


def fail():
    raise Boom("dependency down")


def ok():
    return "ok"


def test_starts_closed_and_passes_through(breaker):
    assert breaker.state is State.CLOSED
    assert breaker.call(ok) == "ok"


def test_opens_after_threshold(breaker):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.state is State.OPEN


def test_open_circuit_fails_fast_without_calling(breaker):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)

    calls = []
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: calls.append(1))
    assert calls == [], "open breaker must not invoke the wrapped callable"


def test_success_resets_failure_count(breaker):
    with pytest.raises(Boom):
        breaker.call(fail)
    with pytest.raises(Boom):
        breaker.call(fail)
    breaker.call(ok)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.state is State.CLOSED, "counter must be consecutive-failures"


def test_half_open_after_recovery_timeout(breaker, clock):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    clock.advance(31)
    assert breaker.state is State.HALF_OPEN


def test_half_open_success_closes(breaker, clock):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    clock.advance(31)
    assert breaker.call(ok) == "ok"
    assert breaker.state is State.CLOSED


def test_half_open_failure_reopens_immediately(breaker, clock):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    clock.advance(31)
    with pytest.raises(Boom):
        breaker.call(fail)
    assert breaker.state is State.OPEN
    # And the recovery window restarts from the new failure.
    assert breaker.snapshot()["retry_after"] == pytest.approx(30.0, abs=0.01)


def test_half_open_limits_probe_calls(breaker, clock):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    clock.advance(31)

    started = []

    def slow_fail():
        started.append(1)
        raise Boom("still down")

    # First probe fails and re-opens; further probes must be refused fast.
    with pytest.raises(Boom):
        breaker.call(slow_fail)
    with pytest.raises(CircuitOpenError):
        breaker.call(slow_fail)
    assert len(started) == 1


def test_uncounted_exception_does_not_trip(clock):
    b = CircuitBreaker(
        name="typed",
        failure_threshold=2,
        counted_exceptions=(Boom,),
        time_source=clock,
    )

    def bad_input():
        raise ValueError("caller bug")

    for _ in range(5):
        with pytest.raises(ValueError):
            b.call(bad_input)
    assert b.state is State.CLOSED


def test_retry_after_decreases(breaker, clock):
    for _ in range(3):
        with pytest.raises(Boom):
            breaker.call(fail)
    assert breaker.snapshot()["retry_after"] == pytest.approx(30.0)
    clock.advance(10)
    assert breaker.snapshot()["retry_after"] == pytest.approx(20.0)


@pytest.mark.anyio
async def test_async_call_path(breaker):
    async def afail():
        raise Boom("async down")

    async def aok():
        return "ok"

    assert await breaker.call_async(aok) == "ok"
    for _ in range(3):
        with pytest.raises(Boom):
            await breaker.call_async(afail)
    assert breaker.state is State.OPEN
    with pytest.raises(CircuitOpenError):
        await breaker.call_async(aok)
