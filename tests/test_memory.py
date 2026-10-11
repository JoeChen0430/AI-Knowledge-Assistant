"""Tiered memory: decay ranking, promotion gate, L0 budget, L1 condensation."""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from aka.db.models import Conversation, MemoryStatus, new_id, utcnow
from aka.memory.extraction import FactCandidate, _evidence_supported, _parse_json
from aka.memory.store import (
    MemoryStore,
    decay_factor,
    normalize_fact,
    reinforcement_factor,
)
from aka.memory.tiers import COMMIT, DROP, REVIEW, TieredMemory, decide_promotion


def vec(text: str, dim: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


class StubRouter:
    """Router stand-in returning canned JSON for extraction/contradiction."""

    def __init__(self, response: str = "[]"):
        self.response = response
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        from aka.llm.router import GenerationResult

        text = self.response(messages) if callable(self.response) else self.response
        return GenerationResult(text=text, model="stub", tier=0, degraded=False)


@pytest.fixture()
def memory(session, settings):
    return TieredMemory(session, settings)


# --------------------------------------------------------------------------
# Ranking primitives
# --------------------------------------------------------------------------

def test_decay_halves_at_half_life():
    created = utcnow() - timedelta(days=30)
    assert decay_factor(created, 30.0) == pytest.approx(0.5, abs=0.01)


def test_decay_is_one_when_fresh():
    assert decay_factor(utcnow(), 30.0) == pytest.approx(1.0, abs=0.001)


def test_reinforcement_grows_and_saturates():
    assert reinforcement_factor(0) == 1.0
    assert reinforcement_factor(5) > reinforcement_factor(1)
    # log-shaped: doubling uses does not double the boost
    assert reinforcement_factor(100) < 2.0


def test_normalize_fact_collapses_variants():
    assert normalize_fact("Joe PREFERS  Python!") == normalize_fact("joe prefers python")


# --------------------------------------------------------------------------
# L2 retrieval ranking
# --------------------------------------------------------------------------

def test_fresh_memory_outranks_stale_one_at_equal_similarity(session, settings):
    store = MemoryStore(session, settings)
    content = "The user works in distributed systems"
    embedding = vec(content)

    old = store.create(
        tenant_id="t1", user_id="u1", content=content + " (old)",
        status=MemoryStatus.ACTIVE, embedding=embedding,
    )
    old.created_at = utcnow() - timedelta(days=120)
    new = store.create(
        tenant_id="t1", user_id="u1", content=content + " (new)",
        status=MemoryStatus.ACTIVE, embedding=embedding,
    )
    session.flush()

    hits = store.search("t1", "u1", embedding, min_score=0.0)
    assert hits[0].item.id == new.id
    assert hits[0].score > hits[1].score


def test_reinforced_memory_survives_age(session, settings):
    store = MemoryStore(session, settings)
    embedding = vec("topic")

    used = store.create(
        tenant_id="t1", user_id="u1", content="used often",
        status=MemoryStatus.ACTIVE, embedding=embedding,
    )
    used.created_at = utcnow() - timedelta(days=40)
    used.use_count = 30
    unused = store.create(
        tenant_id="t1", user_id="u1", content="never used",
        status=MemoryStatus.ACTIVE, embedding=embedding,
    )
    unused.created_at = utcnow() - timedelta(days=40)
    session.flush()

    hits = store.search("t1", "u1", embedding, min_score=0.0)
    assert hits[0].item.id == used.id


def test_search_excludes_pending_and_rejected(session, settings):
    store = MemoryStore(session, settings)
    embedding = vec("secret")
    store.create(tenant_id="t1", user_id="u1", content="pending fact",
                 status=MemoryStatus.PENDING, embedding=embedding)
    store.create(tenant_id="t1", user_id="u1", content="rejected fact",
                 status=MemoryStatus.REJECTED, embedding=embedding)
    session.flush()
    assert store.search("t1", "u1", embedding, min_score=0.0) == []


def test_search_is_tenant_scoped(session, settings):
    store = MemoryStore(session, settings)
    embedding = vec("shared topic")
    store.create(tenant_id="t2", user_id="u1", content="other tenant fact",
                 status=MemoryStatus.ACTIVE, embedding=embedding)
    session.flush()
    assert store.search("t1", "u1", embedding, min_score=0.0) == []


def test_prune_keeps_reinforced_memories(session, settings):
    store = MemoryStore(session, settings)
    ancient = store.create(tenant_id="t1", user_id="u1", content="ancient unused",
                           status=MemoryStatus.ACTIVE, embedding=vec("a"))
    ancient.created_at = utcnow() - timedelta(days=900)
    kept = store.create(tenant_id="t1", user_id="u1", content="ancient but used",
                        status=MemoryStatus.ACTIVE, embedding=vec("b"))
    kept.created_at = utcnow() - timedelta(days=900)
    kept.use_count = 3
    session.flush()

    assert store.prune_decayed("t1", "u1") == 1
    remaining = [m.content for m in store.live_items("t1", "u1")]
    assert remaining == ["ancient but used"]


# --------------------------------------------------------------------------
# Promotion gate
# --------------------------------------------------------------------------

def test_high_confidence_commits(settings):
    d = decide_promotion(FactCandidate("x", confidence=0.95), settings=settings)
    assert d.action == COMMIT
    assert d.status is MemoryStatus.ACTIVE


def test_mid_confidence_goes_to_review(settings):
    d = decide_promotion(FactCandidate("x", confidence=0.6), settings=settings)
    assert d.action == REVIEW
    assert d.status is MemoryStatus.PENDING


def test_low_confidence_dropped(settings):
    d = decide_promotion(FactCandidate("x", confidence=0.1), settings=settings)
    assert d.action == DROP


def test_duplicate_dropped_regardless_of_confidence(settings):
    existing = object()
    d = decide_promotion(
        FactCandidate("x", confidence=0.99), settings=settings, duplicate=existing
    )
    assert d.action == DROP
    assert "already stored" in d.reason


def test_contradiction_forces_review_even_at_high_confidence(settings):
    """The rule that stops the assistant silently overwriting the user."""
    conflict = object()
    d = decide_promotion(
        FactCandidate("x", confidence=0.99), settings=settings, conflict=conflict
    )
    assert d.action == REVIEW
    assert d.conflicts_with is conflict


def test_promote_persists_pending_item(session, settings, memory):
    item, decision = memory.promote(
        FactCandidate("The user lives in Boston", confidence=0.6),
        tenant_id="t1", user_id="u1", router=StubRouter(), embedding=vec("boston"),
    )
    assert decision.action == REVIEW
    assert item.status is MemoryStatus.PENDING


def test_promote_supersedes_on_committed_conflict(session, settings, memory):
    store = memory.store
    old = store.create(
        tenant_id="t1", user_id="u1", content="The user lives in Boston",
        status=MemoryStatus.ACTIVE, embedding=vec("city"),
    )
    session.flush()

    router = StubRouter('{"contradicts": true, "confidence": 0.9, "reason": "different city"}')
    item, decision = memory.promote(
        FactCandidate("The user lives in Seattle", confidence=0.99),
        tenant_id="t1", user_id="u1", router=router, embedding=vec("city"),
    )
    # High confidence but contradicting -> review, and the old fact stays live
    # until a human decides.
    assert decision.action == REVIEW
    assert old.status is MemoryStatus.ACTIVE


def test_promote_deduplicates(session, settings, memory):
    memory.store.create(
        tenant_id="t1", user_id="u1", content="The user likes Python",
        status=MemoryStatus.ACTIVE, embedding=vec("python"),
    )
    session.flush()
    item, decision = memory.promote(
        FactCandidate("the user likes python!", confidence=0.99),
        tenant_id="t1", user_id="u1", router=StubRouter(), embedding=vec("python"),
    )
    assert decision.action == DROP
    assert item is None


# --------------------------------------------------------------------------
# L0 / L1
# --------------------------------------------------------------------------

def _conversation(session, settings) -> str:
    convo = Conversation(id=new_id(), tenant_id="t1", user_id="u1", title="t")
    session.add(convo)
    session.flush()
    return convo.id


def test_l0_respects_turn_limit(session, settings, memory):
    cid = _conversation(session, settings)
    settings.L0_MAX_TURNS = 2
    for i in range(10):
        memory.append_message(
            conversation_id=cid, tenant_id="t1", role="user", content=f"message {i}"
        )
    kept = memory.recent_messages(cid)
    assert len(kept) <= settings.L0_MAX_TURNS * 2
    assert kept[-1].content == "message 9", "must keep the newest turn"


def test_l0_respects_token_budget(session, settings, memory):
    cid = _conversation(session, settings)
    settings.L0_TOKEN_BUDGET = 50
    for i in range(6):
        memory.append_message(
            conversation_id=cid, tenant_id="t1", role="user", content="word " * 100
        )
    kept = memory.recent_messages(cid)
    assert len(kept) == 1, "budget must trim to the newest message"


def test_l1_condenses_messages_outside_the_window(session, settings, memory):
    cid = _conversation(session, settings)
    settings.L0_MAX_TURNS = 1
    for i in range(8):
        memory.append_message(
            conversation_id=cid, tenant_id="t1", role="user", content=f"turn {i}"
        )

    router = StubRouter("The user discussed turns 0 through 5.")
    summary = memory.condense_if_needed(cid, router)

    assert summary is not None
    assert "turns 0 through 5" in summary.summary
    assert summary.covers_through_seq < 7


def test_l1_condense_is_incremental(session, settings, memory):
    cid = _conversation(session, settings)
    settings.L0_MAX_TURNS = 1
    for i in range(6):
        memory.append_message(conversation_id=cid, tenant_id="t1", role="user", content=f"t{i}")

    router = StubRouter("summary v1")
    first = memory.condense_if_needed(cid, router)
    calls_after_first = router.calls

    # No new messages -> no second summarization call.
    second = memory.condense_if_needed(cid, router)
    assert router.calls == calls_after_first
    assert second.id == first.id


def test_l1_failure_preserves_previous_summary(session, settings, memory):
    cid = _conversation(session, settings)
    settings.L0_MAX_TURNS = 1
    for i in range(6):
        memory.append_message(conversation_id=cid, tenant_id="t1", role="user", content=f"t{i}")
    memory.condense_if_needed(cid, StubRouter("good summary"))

    for i in range(6, 12):
        memory.append_message(conversation_id=cid, tenant_id="t1", role="user", content=f"t{i}")

    class BrokenRouter:
        def generate(self, *a, **k):
            raise RuntimeError("model down")

    result = memory.condense_if_needed(cid, BrokenRouter())
    assert result is not None
    assert "good summary" in result.summary


def test_build_context_accounts_tokens(session, settings, memory):
    cid = _conversation(session, settings)
    memory.append_message(conversation_id=cid, tenant_id="t1", role="user", content="hello there")
    memory.store.create(
        tenant_id="t1", user_id="u1", content="The user is a backend engineer",
        status=MemoryStatus.ACTIVE, embedding=vec("engineer"),
    )
    session.flush()

    ctx = memory.build_context(
        tenant_id="t1", user_id="u1", conversation_id=cid, query_vector=vec("engineer")
    )
    assert ctx.l0_tokens > 0
    assert ctx.l2_tokens > 0
    assert ctx.total_tokens == ctx.l0_tokens + ctx.l1_tokens + ctx.l2_tokens
    assert ctx.facts[0].item.use_count == 1, "retrieval must reinforce"


def test_ensure_conversation_rejects_cross_tenant_id(session, settings, memory):
    convo = Conversation(id=new_id(), tenant_id="other-tenant", user_id="u9", title="theirs")
    session.add(convo)
    session.flush()

    result = memory.ensure_conversation(
        conversation_id=convo.id, tenant_id="t1", user_id="u1"
    )
    assert result.tenant_id == "t1"
    assert result.id != convo.id, "must not hand back another tenant's conversation"


# --------------------------------------------------------------------------
# Extraction hardening
# --------------------------------------------------------------------------

def test_parse_json_handles_fenced_output():
    assert _parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_parse_json_salvages_surrounding_prose():
    assert _parse_json('Sure! Here you go: {"contradicts": false} Hope that helps.') == {
        "contradicts": False
    }


def test_parse_json_returns_none_on_garbage():
    assert _parse_json("I cannot help with that.") is None


def test_evidence_must_appear_in_the_turn():
    fabricated = FactCandidate("x", evidence="I own three yachts")
    assert not _evidence_supported(fabricated, "I like writing Python on weekends")

    grounded = FactCandidate("x", evidence="I like writing Python")
    assert _evidence_supported(grounded, "I like writing Python on weekends")
