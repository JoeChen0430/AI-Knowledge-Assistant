"""Degradation-chain tests using a fake Ollama backend."""

from __future__ import annotations

import time

import pytest

from aka.llm.breaker import reset_all_breakers
from aka.llm.client import ChatMessage, Completion, LLMTimeout, LLMUnavailable
from aka.llm.router import AllModelsUnavailable, ModelRouter


class FakeClient:
    """Scriptable stand-in for OllamaClient."""

    def __init__(self, behaviour: dict[str, object]):
        self.behaviour = behaviour
        self.calls: list[str] = []

    def chat(self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2):
        self.calls.append(model)
        outcome = self.behaviour.get(model, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(model)
        if outcome == "empty":
            return Completion(text="   ", model=model)
        return Completion(text=f"answer from {model}", model=model, tokens_in=10, tokens_out=20)

    def chat_stream(self, messages, *, model, timeout=None, max_tokens=None, temperature=0.2):
        self.calls.append(model)
        outcome = self.behaviour.get(model, "ok")
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "empty":
            return iter([])
        return iter([f"tok-{model}-1", f"tok-{model}-2"])

    def ping(self):
        return True


@pytest.fixture()
def make_router(settings):
    def _make(behaviour, chain=("primary", "fallback-a", "fallback-b")):
        reset_all_breakers()
        settings.LLM_MODEL = chain[0]
        settings.LLM_FALLBACK_MODELS = list(chain[1:])
        settings.BREAKER_FAILURE_THRESHOLD = 2
        client = FakeClient(behaviour)
        return ModelRouter(settings=settings, client=client), client

    return _make


MSG = [ChatMessage(role="user", content="hi")]


def test_primary_success_is_not_degraded(make_router):
    router, client = make_router({})
    result = router.generate(MSG)
    assert result.model == "primary"
    assert result.tier == 0
    assert result.degraded is False
    assert client.calls == ["primary"]


def test_falls_back_when_primary_unavailable(make_router):
    router, client = make_router({"primary": LLMUnavailable("connection refused")})
    result = router.generate(MSG)
    assert result.model == "fallback-a"
    assert result.tier == 1
    assert result.degraded is True
    assert "primary" in result.degrade_reason
    assert client.calls == ["primary", "fallback-a"]


def test_walks_the_whole_chain(make_router):
    router, client = make_router(
        {"primary": LLMTimeout("slow"), "fallback-a": LLMUnavailable("down")}
    )
    result = router.generate(MSG)
    assert result.model == "fallback-b"
    assert result.tier == 2
    assert len(result.attempts) == 2


def test_all_tiers_down_raises(make_router):
    router, _ = make_router(
        {
            "primary": LLMUnavailable("down"),
            "fallback-a": LLMUnavailable("down"),
            "fallback-b": LLMUnavailable("down"),
        }
    )
    with pytest.raises(AllModelsUnavailable) as exc:
        router.generate(MSG)
    assert len(exc.value.attempts) == 3


def test_empty_completion_counts_as_failure(make_router):
    router, client = make_router({"primary": "empty"})
    result = router.generate(MSG)
    assert result.model == "fallback-a"


def test_breaker_short_circuits_repeated_primary_failures(make_router):
    router, client = make_router({"primary": LLMUnavailable("down")})

    for _ in range(2):
        router.generate(MSG)
    calls_before = len(client.calls)

    # Breaker for 'primary' is now open: the next request must skip it.
    result = router.generate(MSG)
    assert result.model == "fallback-a"
    assert "primary" not in client.calls[calls_before:], "open breaker must not call the backend"


def test_deadline_prevents_starting_a_doomed_call(make_router):
    router, client = make_router({"primary": LLMUnavailable("down")})
    # Deadline already effectively expired.
    with pytest.raises(AllModelsUnavailable):
        router.generate(MSG, deadline=time.monotonic() + 0.2)
    assert client.calls == [], "no attempt should start with no budget left"


def test_call_timeout_clamped_to_remaining_budget(make_router):
    router, _ = router_and_timeout = make_router({})
    settings = router.settings
    settings.LLM_TIMEOUT_S = 90.0
    # 20s left, tier 0 cap is 90s -> must clamp to just under the remaining budget.
    assert router._call_timeout(20.0, 0) == pytest.approx(19.5)
    # Fallback tiers get a tighter cap so they cannot eat the whole budget.
    assert router._call_timeout(300.0, 1) == pytest.approx(45.0)


def test_stream_falls_back_before_first_token(make_router):
    router, client = make_router({"primary": LLMUnavailable("down")})
    tokens, model, tier = router.stream(MSG)
    assert model == "fallback-a"
    assert tier == 1
    assert list(tokens) == ["tok-fallback-a-1", "tok-fallback-a-2"]


def test_stream_empty_is_treated_as_failure(make_router):
    router, _ = make_router({"primary": "empty"})
    tokens, model, _ = router.stream(MSG)
    assert model == "fallback-a"


def test_health_reports_breaker_state(make_router):
    router, _ = make_router({"primary": LLMUnavailable("down")})
    for _ in range(2):
        router.generate(MSG)
    health = router.health()
    states = {b["name"]: b["state"] for b in health["breakers"]}
    assert states["llm:primary"] == "open"
    assert states["llm:fallback-a"] == "closed"
