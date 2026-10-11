"""Grounding review band, feedback overrides, and threshold calibration."""

from __future__ import annotations

import pytest

from aka.hitl import approvals, audit
from aka.hitl.calibration import (
    ANSWER,
    MEMORY,
    Labelled,
    Thresholds,
    calibrate,
    labelled_decisions,
    replay,
    verdict_for,
)
from aka.hitl.policy import Action, PolicyEngine, Proposal, Verdict, contradicts


@pytest.fixture()
def engine(settings):
    return PolicyEngine(settings)


def answer(grounding, **kw):
    return Proposal(
        action=Action.ANSWER_RELEASE, tenant_id="t1", subject_id=kw.pop("subject_id", "m1"),
        grounding=grounding, has_citations=True, **kw,
    )


def test_grounding_bands(engine, settings):
    settings.ANSWER_MIN_GROUNDING, settings.ANSWER_REVIEW_GROUNDING = 0.15, 0.40
    assert engine.evaluate(answer(0.05)).verdict is Verdict.DENY
    middle = engine.evaluate(answer(0.25))
    assert middle.verdict is Verdict.REVIEW and middle.rule == "answer_release.low_grounding"
    assert middle.confidence == pytest.approx(0.25), "the signal is recorded for calibration"
    assert engine.evaluate(answer(0.65)).verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    "system, approved, expected",
    [
        ("allow", True, False),
        ("allow", False, True),
        ("review", True, True),
        ("review", False, False),
        ("deny", True, True),
        ("deny", False, False),
        ("expired", True, False),
    ],
)
def test_contradicts_truth_table(system, approved, expected):
    assert contradicts(system, approved) is expected


def test_feedback_on_auto_allowed_item_is_an_override(session, engine):
    p = answer(0.9, subject_id="msg-1")
    approvals.record_auto_decision(session, proposal=p, decision=engine.evaluate(p))

    res = approvals.record_feedback(
        session, tenant_id="t1", subject_id="msg-1", approved=False, actor_id="u1"
    )
    assert res.was_override is True and res.system_verdict == "allow"
    stats = audit.override_stats(session, tenant_id="t1")
    assert (stats.total_human_decisions, stats.overrides) == (1, 1)


def test_feedback_agreeing_with_the_engine_is_not_an_override(session, engine):
    p = answer(0.9, subject_id="msg-2")
    approvals.record_auto_decision(session, proposal=p, decision=engine.evaluate(p))
    res = approvals.record_feedback(
        session, tenant_id="t1", subject_id="msg-2", approved=True, actor_id="u1"
    )
    assert res.was_override is False


def test_feedback_on_a_queued_item_resolves_the_approval(session, settings, engine):
    p = answer(0.25, subject_id="msg-3")
    approval = approvals.open_approval(session, proposal=p, decision=engine.evaluate(p), settings=settings)
    res = approvals.record_feedback(
        session, tenant_id="t1", subject_id="msg-3", approved=True, actor_id="u1"
    )
    assert res.approval is not None and res.approval.id == approval.id
    assert approvals.pending(session, tenant_id="t1") == []


def test_feedback_needs_an_automatic_decision(session):
    with pytest.raises(approvals.ApprovalError):
        approvals.record_feedback(
            session, tenant_id="t1", subject_id="never-decided", approved=True, actor_id="u1"
        )


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

CURRENT = Thresholds(memory_auto=0.80, memory_reject=0.35, answer_min=0.15, answer_review=0.40)


def test_verdict_for_mirrors_the_policy_engine(engine, settings):
    t = Thresholds.from_settings(settings)
    for g in (0.0, 0.1, 0.2, 0.3, 0.5, 0.9):
        assert verdict_for(ANSWER, g, t) == engine.evaluate(answer(g)).verdict.value
    for c in (0.1, 0.4, 0.7, 0.85):
        p = Proposal(action=Action.MEMORY_WRITE, tenant_id="t1", subject_id="x", confidence=c)
        assert verdict_for(MEMORY, c, t) == engine.evaluate(p).verdict.value


def _labels(family, rows):
    return [Labelled(family, f"{family}.x", s, ok, "review") for s, ok in rows]


def test_replay_counts_overrides_and_review_load():
    items = _labels(MEMORY, [(0.6, True), (0.7, True), (0.5, False), (0.9, True)])
    r = replay(items, CURRENT)
    # 0.6 and 0.7 approved but queued -> 2 overrides; 0.5 rejected while queued
    # agrees; 0.9 auto-allowed and approved agrees.
    assert (r.overrides, r.queued, r.labelled) == (2, 3, 4)


def test_calibration_moves_the_auto_commit_threshold_down_when_humans_approve():
    # Humans approve everything above 0.6 and reject what is below it.
    rows = [(0.62 + 0.01 * i, True) for i in range(15)] + [(0.40 + 0.01 * i, False) for i in range(10)]
    result = calibrate(_labels(MEMORY, rows), CURRENT)
    assert result.recommended.memory_auto <= 0.65
    assert result.after["memory_write"]["override_rate"] < result.before["memory_write"]["override_rate"]
    assert result.recommended.answer_min == CURRENT.answer_min, "too few answer labels to move"


def test_calibration_keeps_thresholds_without_enough_labels():
    result = calibrate(_labels(MEMORY, [(0.6, True)] * 5), CURRENT)
    assert result.recommended == CURRENT
    assert any("labels" in note for note in result.notes)


def test_labelled_decisions_join_human_verdicts_to_signals(session, settings, engine):
    for i, (g, ok) in enumerate([(0.25, True), (0.30, False), (0.9, False)]):
        p = answer(g, subject_id=f"m{i}")
        d = engine.evaluate(p)
        if d.verdict is Verdict.REVIEW:
            approvals.open_approval(session, proposal=p, decision=d, settings=settings)
        else:
            approvals.record_auto_decision(session, proposal=p, decision=d)
        approvals.record_feedback(session, tenant_id="t1", subject_id=f"m{i}", approved=ok, actor_id="u1")

    items = labelled_decisions(session, tenant_id="t1")
    assert sorted((round(i.signal, 2), i.approved, i.system_verdict) for i in items) == [
        (0.25, True, "review"),
        (0.30, False, "review"),
        (0.90, False, "allow"),
    ]
