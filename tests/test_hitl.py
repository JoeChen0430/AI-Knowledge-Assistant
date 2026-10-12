"""Policy engine, approval queue, audit trail, undo."""

from __future__ import annotations

from datetime import timedelta

import pytest

from aka.db.models import ActorType, ApprovalStatus, utcnow
from aka.hitl import approvals, audit
from aka.hitl.policy import Action, Decision, PolicyEngine, Proposal, Verdict


@pytest.fixture()
def engine(settings):
    return PolicyEngine(settings)


def proposal(action=Action.MEMORY_WRITE, **kw):
    return Proposal(action=action, tenant_id="t1", subject_id=kw.pop("subject_id", "s1"), **kw)


# --------------------------------------------------------------------------
# Policy rules
# --------------------------------------------------------------------------

def test_high_confidence_memory_write_allowed(engine):
    d = engine.evaluate(proposal(confidence=0.95))
    assert d.verdict is Verdict.ALLOW


def test_uncertain_memory_write_goes_to_review(engine):
    d = engine.evaluate(proposal(confidence=0.55))
    assert d.verdict is Verdict.REVIEW
    assert d.requires_human


def test_low_confidence_memory_write_denied(engine):
    d = engine.evaluate(proposal(confidence=0.1))
    assert d.verdict is Verdict.DENY


def test_contradiction_overrides_high_confidence(engine):
    d = engine.evaluate(proposal(confidence=0.99, conflicts_with="mem-42"))
    assert d.verdict is Verdict.REVIEW
    assert d.rule == "memory_write.contradiction"


def test_ungrounded_answer_denied(engine):
    d = engine.evaluate(
        proposal(Action.ANSWER_RELEASE, grounding=0.05, has_citations=True)
    )
    assert d.verdict is Verdict.DENY
    assert "grounding" in d.reason


def test_answer_without_citations_denied(engine):
    d = engine.evaluate(
        proposal(Action.ANSWER_RELEASE, grounding=0.9, has_citations=False)
    )
    assert d.verdict is Verdict.DENY


def test_grounded_answer_allowed(engine):
    d = engine.evaluate(proposal(Action.ANSWER_RELEASE, grounding=0.8, has_citations=True))
    assert d.verdict is Verdict.ALLOW


def test_output_flag_forces_review_even_when_grounded(engine):
    d = engine.evaluate(
        proposal(
            Action.ANSWER_RELEASE, grounding=0.9, has_citations=True,
            output_flags=["system_prompt_leak"],
        )
    )
    assert d.verdict is Verdict.REVIEW


def test_secret_in_document_quarantines(engine):
    d = engine.evaluate(proposal(Action.DOCUMENT_INGEST, scan_severity="high"))
    assert d.verdict is Verdict.REVIEW
    assert d.rule == "document_ingest.secret_detected"


def test_clean_document_ingests(engine):
    d = engine.evaluate(proposal(Action.DOCUMENT_INGEST, scan_severity="low"))
    assert d.verdict is Verdict.ALLOW


def test_injection_flagged_document_reviewed(engine):
    d = engine.evaluate(proposal(Action.DOCUMENT_INGEST, injection_flagged=True))
    assert d.verdict is Verdict.REVIEW


def test_human_initiated_soft_delete_allowed(engine):
    d = engine.evaluate(
        proposal(Action.DOCUMENT_DELETE, actor_is_human=True, irreversible=False)
    )
    assert d.verdict is Verdict.ALLOW


def test_machine_initiated_delete_reviewed(engine):
    d = engine.evaluate(
        proposal(Action.DOCUMENT_DELETE, actor_is_human=False, irreversible=False)
    )
    assert d.verdict is Verdict.REVIEW


def test_irreversible_delete_always_reviewed(engine):
    d = engine.evaluate(
        proposal(Action.DOCUMENT_DELETE, actor_is_human=True, irreversible=True)
    )
    assert d.verdict is Verdict.REVIEW


def test_broken_rule_fails_closed_to_review(engine, monkeypatch):
    """A governance layer that no-ops when broken is worse than none."""
    from aka.hitl import policy

    def exploding_rule(p, s):
        raise ZeroDivisionError("bug in rule")

    monkeypatch.setitem(policy._RULES, Action.MEMORY_WRITE, exploding_rule)
    d = engine.evaluate(proposal(confidence=0.99))
    assert d.verdict is Verdict.REVIEW
    assert "rule_error" in d.rule


def test_disabled_engine_is_loud_but_permissive(settings):
    settings.HITL_ENABLED = False
    d = PolicyEngine(settings).evaluate(proposal(confidence=0.01))
    assert d.verdict is Verdict.ALLOW
    assert d.rule == "hitl.disabled"


# --------------------------------------------------------------------------
# Approval queue
# --------------------------------------------------------------------------

def _queue_one(session, settings, engine, **kw):
    p = proposal(**kw)
    d = engine.evaluate(p)
    return approvals.open_approval(session, proposal=p, decision=d, settings=settings), d


def test_open_approval_records_system_verdict(session, settings, engine):
    approval, decision = _queue_one(session, settings, engine, confidence=0.6)
    assert approval.status is ApprovalStatus.PENDING
    trail = audit.trail_for(session, tenant_id="t1", subject_id="s1")
    assert len(trail) == 1
    assert trail[0].actor_type is ActorType.SYSTEM


def test_resolve_approves_and_audits(session, settings, engine):
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    res = approvals.resolve(
        session, approval_id=approval.id, tenant_id="t1", approved=True, resolved_by="reviewer-1"
    )
    assert res.approval.status is ApprovalStatus.APPROVED
    assert res.was_override is True

    trail = audit.trail_for(session, tenant_id="t1", subject_id="s1")
    assert [e.actor_type for e in trail] == [ActorType.SYSTEM, ActorType.HUMAN]
    assert trail[1].overrides_audit_id == trail[0].id


def test_rejection_is_not_an_override(session, settings, engine):
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    res = approvals.resolve(
        session, approval_id=approval.id, tenant_id="t1", approved=False, resolved_by="r1"
    )
    assert res.was_override is False
    trail = audit.trail_for(session, tenant_id="t1", subject_id="s1")
    assert trail[1].overrides_audit_id is None


def test_cannot_resolve_twice(session, settings, engine):
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    approvals.resolve(
        session, approval_id=approval.id, tenant_id="t1", approved=True, resolved_by="r1"
    )
    with pytest.raises(approvals.ApprovalError, match="already"):
        approvals.resolve(
            session, approval_id=approval.id, tenant_id="t1", approved=False, resolved_by="r2"
        )


def test_cannot_resolve_another_tenants_approval(session, settings, engine):
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    with pytest.raises(approvals.ApprovalError, match="not found"):
        approvals.resolve(
            session, approval_id=approval.id, tenant_id="attacker", approved=True, resolved_by="x"
        )


def test_expired_approval_cannot_be_resolved(session, settings, engine):
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    approval.expires_at = utcnow() - timedelta(hours=1)
    session.flush()
    with pytest.raises(approvals.ApprovalError, match="expired"):
        approvals.resolve(
            session, approval_id=approval.id, tenant_id="t1", approved=True, resolved_by="r1"
        )


def test_expiry_does_not_apply_the_action(session, settings, engine):
    """Expiry must mean 'not done', never 'done by default'."""
    approval, _ = _queue_one(session, settings, engine, confidence=0.6)
    approval.expires_at = utcnow() - timedelta(hours=1)
    session.flush()

    assert approvals.expire_stale(session, "t1") == 1
    session.refresh(approval)
    assert approval.status is ApprovalStatus.EXPIRED

    decisions = [e.decision for e in audit.trail_for(session, tenant_id="t1", subject_id="s1")]
    assert "expired" in decisions
    assert "approved" not in decisions


def test_pending_listing_is_tenant_scoped(session, settings, engine):
    _queue_one(session, settings, engine, confidence=0.6)
    assert len(approvals.pending(session, tenant_id="t1")) == 1
    assert approvals.pending(session, tenant_id="t2") == []


# --------------------------------------------------------------------------
# Override rate
# --------------------------------------------------------------------------

def test_override_rate_is_measurable(session, settings, engine):
    """The number that drives threshold tuning."""
    for i in range(10):
        p = Proposal(action=Action.MEMORY_WRITE, tenant_id="t1", subject_id=f"m{i}", confidence=0.6)
        d = engine.evaluate(p)
        a = approvals.open_approval(session, proposal=p, decision=d, settings=settings)
        # 3 of 10 queued items were things a human would have allowed.
        approvals.resolve(
            session, approval_id=a.id, tenant_id="t1", approved=i < 3, resolved_by="r1"
        )

    stats = audit.override_stats(session, tenant_id="t1")
    assert stats.total_human_decisions == 10
    assert stats.overrides == 3
    assert stats.rate == pytest.approx(0.3)


def test_override_stats_can_filter_by_rule(session, settings, engine):
    p = Proposal(action=Action.MEMORY_WRITE, tenant_id="t1", subject_id="m1", confidence=0.6)
    d = engine.evaluate(p)
    a = approvals.open_approval(session, proposal=p, decision=d, settings=settings)
    approvals.resolve(session, approval_id=a.id, tenant_id="t1", approved=True, resolved_by="r1")

    matching = audit.override_stats(session, tenant_id="t1", rule="memory_write.uncertain")
    assert matching.overrides == 1
    other = audit.override_stats(session, tenant_id="t1", rule="memory_write.contradiction")
    assert other.total_human_decisions == 0


def test_auto_decisions_are_audited(session, settings, engine):
    p = Proposal(action=Action.MEMORY_WRITE, tenant_id="t1", subject_id="m1", confidence=0.99)
    d = engine.evaluate(p)
    assert d.verdict is Verdict.ALLOW
    approvals.record_auto_decision(session, proposal=p, decision=d)

    trail = audit.trail_for(session, tenant_id="t1", subject_id="m1")
    assert len(trail) == 1
    assert trail[0].payload["auto"] is True


def test_rule_breakdown_groups_by_rule(session, settings, engine):
    for conf in (0.99, 0.6, 0.6):
        p = Proposal(action=Action.MEMORY_WRITE, tenant_id="t1", subject_id=f"m{conf}", confidence=conf)
        d = engine.evaluate(p)
        if d.requires_human:
            approvals.open_approval(session, proposal=p, decision=d, settings=settings)
        else:
            approvals.record_auto_decision(session, proposal=p, decision=d)

    breakdown = {row["rule"]: row for row in audit.rule_breakdown(session, tenant_id="t1")}
    assert "memory_write.high_confidence" in breakdown
    assert "memory_write.uncertain" in breakdown


# --------------------------------------------------------------------------
# Undo
# --------------------------------------------------------------------------

def test_undo_token_round_trip(session, settings):
    token = approvals.issue_undo(
        session, tenant_id="t1", action="document_delete",
        payload={"rel_path": "a.pdf"}, settings=settings,
    )
    session.commit()
    consumed = approvals.consume_undo(session, token_id=token.id, tenant_id="t1")
    assert consumed.payload["rel_path"] == "a.pdf"
    assert consumed.consumed_at is not None


def test_undo_token_is_single_use(session, settings):
    token = approvals.issue_undo(
        session, tenant_id="t1", action="document_delete", payload={}, settings=settings
    )
    session.commit()
    approvals.consume_undo(session, token_id=token.id, tenant_id="t1")
    with pytest.raises(approvals.ApprovalError, match="already used"):
        approvals.consume_undo(session, token_id=token.id, tenant_id="t1")


def test_expired_undo_token_rejected(session, settings):
    token = approvals.issue_undo(
        session, tenant_id="t1", action="document_delete", payload={}, settings=settings
    )
    token.expires_at = utcnow() - timedelta(minutes=1)
    session.commit()
    with pytest.raises(approvals.ApprovalError, match="expired"):
        approvals.consume_undo(session, token_id=token.id, tenant_id="t1")


def test_undo_token_is_tenant_scoped(session, settings):
    token = approvals.issue_undo(
        session, tenant_id="t1", action="document_delete", payload={}, settings=settings
    )
    session.commit()
    with pytest.raises(approvals.ApprovalError, match="not found"):
        approvals.consume_undo(session, token_id=token.id, tenant_id="attacker")
