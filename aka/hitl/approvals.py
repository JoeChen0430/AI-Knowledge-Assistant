"""Approval queue and reversible actions.

Two guarantees this module provides:

* Nothing queued is lost or silently applied. An approval either gets a human
  verdict, or it expires — and expiry is an explicit, audited outcome, not a
  timeout that quietly lets the action through.
* Every destructive action issues an undo token valid for a bounded window, so
  "reversible" is a property of the system rather than of the operator's luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from aka.db.models import (
    ActorType,
    Approval,
    ApprovalStatus,
    AuditLog,
    UndoToken,
    new_id,
    utcnow,
)
from aka.hitl import audit
from aka.hitl.policy import Decision, Proposal, Verdict, contradicts
from aka.obs.logging import get_logger
from aka.obs.metrics import hitl_overrides, hitl_queue_depth, hitl_resolution_seconds
from aka.obs.telemetry import current_trace_id
from aka.settings import Settings, get_settings

log = get_logger("aka.hitl.approvals")


class ApprovalError(RuntimeError):
    pass


@dataclass
class Resolution:
    approval: Approval | None  # None for feedback on an item that was never queued
    was_override: bool
    audit_id: str
    system_verdict: str = Verdict.REVIEW.value


def open_approval(
    session: Session,
    *,
    proposal: Proposal,
    decision: Decision,
    requested_by: str | None = None,
    settings: Settings | None = None,
) -> Approval:
    """Queue a decision for a human and record the machine verdict that led here."""
    settings = settings or get_settings()

    system_entry = audit.record(
        session,
        tenant_id=proposal.tenant_id,
        action=proposal.action.value,
        decision=decision.verdict.value,
        actor_type=ActorType.SYSTEM,
        subject_type=proposal.subject_type or proposal.action.value,
        subject_id=proposal.subject_id,
        rule=decision.rule,
        payload={"reason": decision.reason, "confidence": decision.confidence},
    )

    approval = Approval(
        id=new_id(),
        tenant_id=proposal.tenant_id,
        kind=proposal.action.value,
        subject_type=proposal.subject_type or proposal.action.value,
        subject_id=proposal.subject_id,
        title=proposal.title[:255] or proposal.action.value,
        reason=decision.reason,
        rule=decision.rule,
        confidence=decision.confidence,
        payload={
            **proposal.payload,
            "review_context": decision.review_context,
            "system_audit_id": system_entry.id,
        },
        status=ApprovalStatus.PENDING,
        requested_by=requested_by,
        expires_at=utcnow() + timedelta(hours=settings.APPROVAL_TTL_HOURS),
        trace_id=current_trace_id(),
    )
    session.add(approval)
    session.flush()

    refresh_queue_gauge(session, proposal.tenant_id)
    log.info("hitl.queued", approval_id=approval.id, kind=approval.kind, rule=decision.rule)
    return approval


def record_auto_decision(
    session: Session, *, proposal: Proposal, decision: Decision
) -> str:
    """Audit an action taken without a human. Returns the audit row id.

    Auto-approvals are logged as deliberately as queued ones; otherwise the
    override rate has no denominator and the audit trail has a hole exactly
    where the automation acted.
    """
    entry = audit.record(
        session,
        tenant_id=proposal.tenant_id,
        action=proposal.action.value,
        decision=decision.verdict.value,
        actor_type=ActorType.SYSTEM,
        subject_type=proposal.subject_type or proposal.action.value,
        subject_id=proposal.subject_id,
        rule=decision.rule,
        payload={"reason": decision.reason, "confidence": decision.confidence, "auto": True},
    )
    return entry.id


def resolve(
    session: Session,
    *,
    approval_id: str,
    tenant_id: str,
    approved: bool,
    resolved_by: str,
    note: str = "",
) -> Resolution:
    """Apply a human verdict.

    Tenant-scoped by query, not by check-after-fetch, so a reviewer cannot
    resolve another tenant's approval by guessing an id.
    """
    approval = session.scalars(
        select(Approval).where(Approval.id == approval_id, Approval.tenant_id == tenant_id)
    ).first()
    if approval is None:
        raise ApprovalError("approval not found")
    if approval.status is not ApprovalStatus.PENDING:
        raise ApprovalError(f"approval already {approval.status.value}")

    if approval.expires_at and approval.expires_at < utcnow():
        expire(session, approval)
        raise ApprovalError("approval expired before resolution")

    approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
    approval.resolved_by = resolved_by
    approval.resolved_at = utcnow()
    approval.resolution_note = note[:2000] or None

    # An override is a human verdict that contradicts what the engine wanted.
    # The engine queued this item, i.e. it declined to auto-approve; a human
    # approving it is therefore an override of that caution.
    system_audit_id = approval.payload.get("system_audit_id")
    system_entry = session.get(AuditLog, system_audit_id) if system_audit_id else None
    system_verdict = system_entry.decision if system_entry else Verdict.REVIEW.value
    was_override = contradicts(system_verdict, approved)

    human_entry = audit.record(
        session,
        tenant_id=tenant_id,
        action=approval.kind,
        decision="approved" if approved else "rejected",
        actor_type=ActorType.HUMAN,
        actor_id=resolved_by,
        subject_type=approval.subject_type,
        subject_id=approval.subject_id,
        rule=approval.rule,
        overrides_audit_id=system_audit_id if was_override else None,
        payload={"note": note[:500], "approval_id": approval.id},
    )

    session.flush()

    hitl_overrides.labels(
        kind=approval.kind, verdict="approved" if approved else "rejected"
    ).inc()
    if approval.resolved_at and approval.created_at:
        created = approval.created_at
        if created.tzinfo is None:
            from datetime import timezone

            created = created.replace(tzinfo=timezone.utc)
        hitl_resolution_seconds.observe((approval.resolved_at - created).total_seconds())

    refresh_queue_gauge(session, tenant_id)
    log.info(
        "hitl.resolved",
        approval_id=approval.id,
        approved=approved,
        by=resolved_by,
        override=was_override,
    )
    return Resolution(
        approval=approval,
        was_override=was_override,
        audit_id=human_entry.id,
        system_verdict=system_verdict,
    )


def record_feedback(
    session: Session,
    *,
    tenant_id: str,
    subject_id: str,
    approved: bool,
    actor_id: str,
    note: str = "",
) -> Resolution:
    """A human verdict on something the engine decided without asking:
    thumbs up/down on a released answer, keep/forget on an auto-committed
    memory.

    This is the other half of the override rate. Queued items only show where
    the engine was too cautious; feedback on auto-approved items shows where it
    was not cautious enough. Without it, raising every threshold would look
    like a perfect score.
    """
    pending_approval = session.scalars(
        select(Approval).where(
            Approval.tenant_id == tenant_id,
            Approval.subject_id == subject_id,
            Approval.status == ApprovalStatus.PENDING,
        )
    ).first()
    if pending_approval is not None:
        # Still queued: this feedback *is* the review.
        return resolve(
            session,
            approval_id=pending_approval.id,
            tenant_id=tenant_id,
            approved=approved,
            resolved_by=actor_id,
            note=note,
        )

    system_entry = audit.latest_system_decision(session, tenant_id=tenant_id, subject_id=subject_id)
    if system_entry is None:
        raise ApprovalError("no automatic decision is recorded for this item")

    already = session.scalars(
        select(AuditLog.id).where(
            AuditLog.tenant_id == tenant_id,
            AuditLog.subject_id == subject_id,
            AuditLog.actor_type == ActorType.HUMAN,
        )
    ).first()
    if already is not None:
        # One verdict per item keeps the override-rate denominator honest.
        raise ApprovalError("a verdict was already recorded for this item")

    was_override = contradicts(system_entry.decision, approved)
    human_entry = audit.record(
        session,
        tenant_id=tenant_id,
        action=system_entry.action,
        decision="approved" if approved else "rejected",
        actor_type=ActorType.HUMAN,
        actor_id=actor_id,
        subject_type=system_entry.subject_type,
        subject_id=subject_id,
        rule=system_entry.rule,
        overrides_audit_id=system_entry.id if was_override else None,
        payload={"note": note[:500], "source": "feedback"},
    )
    hitl_overrides.labels(
        kind=system_entry.action, verdict="approved" if approved else "rejected"
    ).inc()
    log.info(
        "hitl.feedback",
        subject=subject_id,
        approved=approved,
        system_verdict=system_entry.decision,
        override=was_override,
    )
    return Resolution(
        approval=None,
        was_override=was_override,
        audit_id=human_entry.id,
        system_verdict=system_entry.decision,
    )


def expire(session: Session, approval: Approval) -> None:
    approval.status = ApprovalStatus.EXPIRED
    approval.resolved_at = utcnow()
    audit.record(
        session,
        tenant_id=approval.tenant_id,
        action=approval.kind,
        decision="expired",
        actor_type=ActorType.SYSTEM,
        subject_type=approval.subject_type,
        subject_id=approval.subject_id,
        rule=approval.rule,
        payload={"approval_id": approval.id},
    )
    session.flush()


def expire_stale(session: Session, tenant_id: str | None = None) -> int:
    """Expire timed-out approvals.

    Expiry means "not done", never "done by default" — the pending action stays
    unapplied. Anything else would make the TTL a way to smuggle changes past
    review by waiting.
    """
    now = utcnow()
    q = select(Approval).where(
        Approval.status == ApprovalStatus.PENDING,
        Approval.expires_at.is_not(None),
        Approval.expires_at < now,
    )
    if tenant_id:
        q = q.where(Approval.tenant_id == tenant_id)

    stale = list(session.scalars(q))
    for approval in stale:
        expire(session, approval)

    if stale:
        log.warning("hitl.expired", count=len(stale))
        refresh_queue_gauge(session, tenant_id or stale[0].tenant_id)
    return len(stale)


def pending(
    session: Session, *, tenant_id: str, kind: str | None = None, limit: int = 50
) -> list[Approval]:
    q = select(Approval).where(
        Approval.tenant_id == tenant_id, Approval.status == ApprovalStatus.PENDING
    )
    if kind:
        q = q.where(Approval.kind == kind)
    return list(session.scalars(q.order_by(Approval.created_at).limit(limit)))


def refresh_queue_gauge(session: Session, tenant_id: str) -> None:
    rows = session.execute(
        select(Approval.kind, func.count(Approval.id))
        .where(Approval.tenant_id == tenant_id, Approval.status == ApprovalStatus.PENDING)
        .group_by(Approval.kind)
    ).all()
    seen = set()
    for kind, count in rows:
        hitl_queue_depth.labels(kind=kind).set(count)
        seen.add(kind)
    # Zero out kinds that just drained, or the gauge stays stale forever.
    for kind in ("memory_write", "document_delete", "document_ingest", "answer_release"):
        if kind not in seen:
            hitl_queue_depth.labels(kind=kind).set(0)


# --------------------------------------------------------------------------
# Undo
# --------------------------------------------------------------------------

def issue_undo(
    session: Session,
    *,
    tenant_id: str,
    action: str,
    payload: dict,
    created_by: str | None = None,
    settings: Settings | None = None,
) -> UndoToken:
    settings = settings or get_settings()
    token = UndoToken(
        id=new_id(),
        tenant_id=tenant_id,
        action=action,
        payload=payload,
        created_by=created_by,
        expires_at=utcnow() + timedelta(hours=settings.UNDO_WINDOW_HOURS),
    )
    session.add(token)
    session.flush()
    log.info("hitl.undo_issued", token_id=token.id, action=action)
    return token


def consume_undo(session: Session, *, token_id: str, tenant_id: str) -> UndoToken:
    """Claim an undo token exactly once.

    The single-statement UPDATE is what makes it exactly-once: two concurrent
    undo requests cannot both see it as unconsumed.
    """
    now = utcnow()
    result = session.execute(
        update(UndoToken)
        .where(
            UndoToken.id == token_id,
            UndoToken.tenant_id == tenant_id,
            UndoToken.consumed_at.is_(None),
            UndoToken.expires_at > now,
        )
        .values(consumed_at=now)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        existing = session.get(UndoToken, token_id)
        if existing is None or existing.tenant_id != tenant_id:
            raise ApprovalError("undo token not found")
        if existing.consumed_at is not None:
            raise ApprovalError("undo token already used")
        raise ApprovalError("undo window has expired")

    session.commit()
    token = session.scalars(
        select(UndoToken).where(UndoToken.id == token_id).execution_options(populate_existing=True)
    ).one()
    log.info("hitl.undo_consumed", token_id=token.id, action=token.action)
    return token
