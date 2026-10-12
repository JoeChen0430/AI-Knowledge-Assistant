"""Fault injection, embedding resilience, and the pipeline under failure."""

from __future__ import annotations

import time
from collections import Counter

import pytest

from aka.llm.client import ChatMessage, LLMTimeout, LLMUnavailable
from aka.llm.embedding import EmbeddingUnavailable, ResilientEmbedder
from aka.llm.fake import FakeOllama
from aka.llm.faults import FaultInjectingClient

DOCS = {
    "hermes.txt": (
        "The Hermes deployment runs on three nodes in the Frankfurt region. "
        "Failover is automatic and takes about ninety seconds."
    ),
    "oncall.txt": "The on-call rotation is weekly. Handover happens every Monday at ten.",
    "budget.txt": "The infrastructure budget for next year is capped at forty thousand euros.",
}
QUESTIONS = [
    "How long does Hermes failover take?",
    "When does the on-call handover happen?",
    "What is the infrastructure budget cap?",
]


# --------------------------------------------------------------------------
# Fault injector
# --------------------------------------------------------------------------

def _embed_outcomes(client, n):
    out = []
    for _ in range(n):
        try:
            client.embed(["x"])
            out.append(True)
        except (LLMTimeout, LLMUnavailable):
            out.append(False)
    return out


def test_fault_injector_is_reproducible_and_hits_its_rate():
    a = FaultInjectingClient(FakeOllama(), rate=0.3, seed=1, timeout_delay_s=0)
    b = FaultInjectingClient(FakeOllama(), rate=0.3, seed=1, timeout_delay_s=0)
    first, second = _embed_outcomes(a, 2000), _embed_outcomes(b, 2000)
    assert first == second, "same seed must fail the same calls"
    assert 0.27 < first.count(False) / 2000 < 0.33
    assert a.fault_summary()["embed.faults"] == first.count(False)


def test_fault_injector_only_hits_its_targets():
    client = FaultInjectingClient(FakeOllama(), rate=1.0, targets={"chat"}, timeout_delay_s=0)
    assert client.embed(["untouched"])
    with pytest.raises((LLMTimeout, LLMUnavailable)):
        client.chat([ChatMessage("user", "hi")], model="m")


def test_production_refuses_fault_injection(monkeypatch):
    from aka.settings import get_settings, reset_settings_cache

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("JWT_SECRET", "a-real-secret-that-is-long-enough-for-hs256")
    monkeypatch.setenv("FAULT_INJECTION_RATE", "0.3")
    reset_settings_cache()
    with pytest.raises(RuntimeError, match="fault injection"):
        get_settings()
    reset_settings_cache()


# --------------------------------------------------------------------------
# Resilient embedder
# --------------------------------------------------------------------------

class Flaky:
    def __init__(self, fail_times: int, exc=LLMUnavailable):
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0

    def embed(self, texts, *, model=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("transient")
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture()
def fast_retry(settings):
    settings.EMBED_RETRY_BACKOFF_S = 0.0
    settings.EMBED_RETRY_ATTEMPTS = 3
    return settings


def test_embedder_retries_transient_failures(fast_retry):
    flaky = Flaky(fail_times=2)
    assert ResilientEmbedder(flaky, fast_retry).embed(["hi"]) == [[1.0, 0.0]]
    assert flaky.calls == 3


def test_embedder_gives_up_after_its_budget(fast_retry):
    flaky = Flaky(fail_times=99)
    with pytest.raises(EmbeddingUnavailable):
        ResilientEmbedder(flaky, fast_retry).embed(["hi"])
    assert flaky.calls == 3


def test_embedder_does_not_retry_bugs(fast_retry):
    flaky = Flaky(fail_times=99, exc=ValueError)
    with pytest.raises(ValueError):
        ResilientEmbedder(flaky, fast_retry).embed(["hi"])
    assert flaky.calls == 1


def test_embedding_breaker_fails_fast_once_open(fast_retry):
    fast_retry.BREAKER_FAILURE_THRESHOLD = 2
    flaky = Flaky(fail_times=99)
    embedder = ResilientEmbedder(flaky, fast_retry)
    with pytest.raises(EmbeddingUnavailable):
        embedder.embed(["a"])
    calls = flaky.calls
    with pytest.raises(EmbeddingUnavailable, match="open"):
        embedder.embed(["b"])
    assert flaky.calls == calls, "an open breaker must not call the backend"


def test_query_vectors_are_cached(settings):
    fake = FakeOllama()
    embedder = ResilientEmbedder(fake, settings)
    embedder.embed_query("How long is failover?")
    embedder.embed_query("  how long is   FAILOVER? ")
    assert fake.embed_calls == 1


# --------------------------------------------------------------------------
# The pipeline under failure
# --------------------------------------------------------------------------

@pytest.fixture()
def corpus(settings, db, session):
    from aka.ingest import vectorstore
    from aka.ingest.pipeline import IncrementalIngestor

    vectorstore.reset_chroma_for_tests()
    for name, body in DOCS.items():
        (settings.DATA_DIR / name).write_text(body, encoding="utf-8")
    IncrementalIngestor(session, settings, client=FakeOllama()).run("t1")
    session.commit()

    settings.RETRIEVAL_MIN_SCORE = 0.05  # bag-of-words fake embeddings
    settings.LLM_MODEL = "big"
    settings.LLM_FALLBACK_MODELS = ["small"]
    settings.EMBED_RETRY_BACKOFF_S = 0.0
    yield settings
    vectorstore.reset_chroma_for_tests()


def make_pipeline(session, settings, client, cache=None):
    from aka.llm.cache import AnswerCache
    from aka.llm.router import ModelRouter
    from aka.rag.pipeline import RagPipeline

    return RagPipeline(
        session,
        settings,
        router=ModelRouter(settings, client=client),
        embedder=ResilientEmbedder(client, settings),
        cache=cache if cache is not None else AnswerCache(),
    )


def test_pipeline_keeps_answering_under_30_percent_faults(corpus, session):
    client = FaultInjectingClient(FakeOllama(), rate=0.3, seed=11, timeout_delay_s=0)
    pipe = make_pipeline(session, corpus, client)

    outcomes = Counter()
    for i in range(240):
        # Distinct wording each time, so the query-vector cache cannot hide
        # embedding faults.
        result = pipe.answer(f"{QUESTIONS[i % 3]} (asked {i:03d})", tenant_id="t1", user_id="u1")
        session.commit()
        outcomes[result.outcome] += 1

    success = (outcomes["answered"] + outcomes["policy_refused"]) / 240
    assert success >= 0.99, outcomes
    faults = client.fault_summary()
    assert faults["chat.faults"] > 30 and faults["embed.faults"] > 30


def test_embedding_outage_falls_back_to_keyword_retrieval(corpus, session):
    client = FaultInjectingClient(FakeOllama(), rate=1.0, targets={"embed"}, timeout_delay_s=0)
    result = make_pipeline(session, corpus, client).answer(QUESTIONS[0], tenant_id="t1", user_id="u1")

    assert result.retrieval_mode == "lexical"
    assert result.evidence and result.evidence[0].source == "hermes.txt"
    assert result.mode == "generated" and result.outcome == "answered"
    assert result.degraded and "keyword" in result.degrade_reason


def test_embedding_outage_serves_a_verified_cached_answer(corpus, session):
    from aka.llm.cache import AnswerCache

    cache = AnswerCache()
    healthy = make_pipeline(session, corpus, FakeOllama(), cache=cache)
    first = healthy.answer(QUESTIONS[0], tenant_id="t1", user_id="u1")
    assert first.policy_verdict == "allow" and len(cache) == 1

    outage = FaultInjectingClient(FakeOllama(), rate=1.0, targets={"embed"}, timeout_delay_s=0)
    second = make_pipeline(session, corpus, outage, cache=cache).answer(
        QUESTIONS[0], tenant_id="t1", user_id="u1"
    )
    assert second.mode == "cached"
    assert second.answer == first.answer
    assert "sources are unchanged" in second.degrade_reason


def test_cached_answer_is_not_served_once_its_evidence_changed(corpus, session):
    from aka.ingest.pipeline import IncrementalIngestor
    from aka.llm.cache import AnswerCache

    cache = AnswerCache()
    make_pipeline(session, corpus, FakeOllama(), cache=cache).answer(
        QUESTIONS[0], tenant_id="t1", user_id="u1"
    )
    (corpus.DATA_DIR / "hermes.txt").write_text(
        "The Hermes deployment moved to Dublin. Failover now takes four minutes.", encoding="utf-8"
    )
    IncrementalIngestor(session, corpus, client=FakeOllama()).run("t1")
    session.commit()

    outage = FaultInjectingClient(FakeOllama(), rate=1.0, targets={"embed"}, timeout_delay_s=0)
    result = make_pipeline(session, corpus, outage, cache=cache).answer(
        QUESTIONS[0], tenant_id="t1", user_id="u1"
    )
    assert result.mode != "cached", "stale evidence must not be served from cache"
    assert "Dublin" in result.answer, "answered from the new evidence"


def test_total_outage_still_shows_matching_passages(corpus, session):
    client = FaultInjectingClient(FakeOllama(), rate=1.0, timeout_delay_s=0)
    result = make_pipeline(session, corpus, client).answer(QUESTIONS[0], tenant_id="t1", user_id="u1")
    assert result.mode == "extractive"
    assert result.outcome == "answered"
    assert "ninety seconds" in result.answer


def test_request_budget_bounds_latency_when_models_hang(corpus, session):
    corpus.REQUEST_BUDGET_S = 2.0
    # Every chat call "hangs" until its own timeout; the budget must cut it off.
    client = FaultInjectingClient(FakeOllama(), rate=1.0, targets={"chat"}, timeout_delay_s=30)
    client._rng.random = lambda: 0.0  # every fault is a timeout
    started = time.monotonic()
    result = make_pipeline(session, corpus, client).answer(QUESTIONS[0], tenant_id="t1", user_id="u1")
    assert time.monotonic() - started < 3.0
    assert result.mode == "extractive"
