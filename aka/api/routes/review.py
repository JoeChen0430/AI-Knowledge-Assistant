"""Human review: the approval queue, memory keep/forget, override statistics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.api.deps import get_principal, get_session, get_settings_dep, require_role
from aka.api.schemas import MemoryFeedbackRequest, ResolveRequest
from aka.db.models import Approval, MemoryItem, MemoryStatus
from aka.hitl import approvals, audit, effects
from aka.hitl.policy import Action
from aka.security.auth import ROLE_REVIEWER, Principal
from aka.settings import Settings

router = APIRouter(tags=["review"])


def _approval_dict(a: Approval) -> dict:
    payload = dict(a.payload or {})
    payload.pop("system_audit_id", None)
    return {
        "id": a.id,
        "kind": a.kind,
        "subject_type": a.subject_type,
        "subject_id": a.subject_id,
        "title": a.title,
        "reason": a.reason,
        "rule": a.rule,
        "confidence": a.confidence,
        "status": a.status.value,
        "context": payload,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
    }


@router.get("/approvals")
def list_approvals(
    kind: str | None = None,
    principal: Principal = Depends(require_role(ROLE_REVIEWER)),
    session: Session = Depends(get_session),
) -> list[dict]:
    approvals.expire_stale(session, principal.tenant_id)
    session.commit()
    return [
        _approval_dict(a)
        for a in approvals.pending(session, tenant_id=principal.tenant_id, kind=kind, limit=200)
    ]


@router.post("/approvals/{approval_id}/resolve")
def resolve_approval(
    approval_id: str,
    body: ResolveRequest,
    principal: Principal = Depends(require_role(ROLE_REVIEWER)),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    try:
        resolution = approvals.resolve(
            session,
            approval_id=approval_id,
            tenant_id=principal.tenant_id,
            approved=body.approved,
            resolved_by=principal.user_id,
            note=body.note,
        )
    except approvals.ApprovalError as exc:
        session.commit()  # keep an expiry recorded by resolve()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    approval = resolution.approval
    applied = effects.apply_verdict(
        session,
        kind=approval.kind,
        tenant_id=principal.tenant_id,
        subject_id=approval.subject_id,
        approved=body.approved,
        context=approval.payload,
        settings=settings,
    )
    session.commit()
    return {
        "approval": _approval_dict(approval),
        "was_override": resolution.was_override,
        "effect": applied,
    }


@router.get("/memory")
def list_memory(
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
) -> list[dict]:
    """What the assistant remembers about the caller, including facts still
    awaiting review."""
    items = session.scalars(
        select(MemoryItem)
        .where(
            MemoryItem.tenant_id == principal.tenant_id,
            MemoryItem.user_id == principal.user_id,
            MemoryItem.status.in_([MemoryStatus.ACTIVE, MemoryStatus.PENDING]),
        )
        .order_by(MemoryItem.created_at.desc())
        .limit(200)
    )
    return [
        {
            "id": m.id,
            "content": m.content,
            "kind": m.kind,
            "confidence": round(m.confidence, 3),
            "status": m.status.value,
            "evidence": m.evidence,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in items
    ]


@router.post("/memory/{memory_id}/feedback")
def memory_feedback(
    memory_id: str,
    body: MemoryFeedbackRequest,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Keep or forget a memory. On an auto-committed one, "forget" is the
    human overriding the engine's decision to store it."""
    item = session.get(MemoryItem, memory_id)
    if item is None or item.tenant_id != principal.tenant_id or item.user_id != principal.user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "memory not found")
    try:
        resolution = approvals.record_feedback(
            session,
            tenant_id=principal.tenant_id,
            subject_id=item.id,
            approved=body.keep,
            actor_id=principal.user_id,
            note=body.note,
        )
    except approvals.ApprovalError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    context = resolution.approval.payload if resolution.approval else {}
    applied = effects.apply_verdict(
        session,
        kind=Action.MEMORY_WRITE.value,
        tenant_id=principal.tenant_id,
        subject_id=item.id,
        approved=body.keep,
        context=context,
        settings=settings,
    )
    session.commit()
    return {"was_override": resolution.was_override, "effect": applied}


@router.get("/stats/overrides")
def override_stats(
    window_days: int = Query(default=30, ge=1, le=365),
    rule: str | None = None,
    principal: Principal = Depends(require_role(ROLE_REVIEWER)),
    session: Session = Depends(get_session),
) -> dict:
    """Fraction of human verdicts that contradicted the engine — overall, and
    per rule, which is the table thresholds are tuned from."""
    overall = audit.override_stats(
        session, tenant_id=principal.tenant_id, window_days=window_days, rule=rule
    )
    rules = sorted(
        {row["rule"] for row in audit.rule_breakdown(session, tenant_id=principal.tenant_id, window_days=window_days)}
    )
    per_rule = {
        r: audit.override_stats(
            session, tenant_id=principal.tenant_id, window_days=window_days, rule=r
        ).as_dict()
        for r in rules
    }
    return {
        **overall.as_dict(),
        "per_rule": {r: s for r, s in per_rule.items() if s["human_decisions"]},
    }


@router.get("/stats/rules")
def rule_stats(
    window_days: int = Query(default=30, ge=1, le=365),
    principal: Principal = Depends(require_role(ROLE_REVIEWER)),
    session: Session = Depends(get_session),
) -> list[dict]:
    return audit.rule_breakdown(session, tenant_id=principal.tenant_id, window_days=window_days)
