"""Append-only audit log, and the override rate derived from it.

The design point: automatic verdicts and the human verdicts that later
contradict them are written to the *same* table, with the human row pointing at
the machine row it overrides. That makes "how often is the policy engine wrong"
a query rather than a guess, which is what lets thresholds be tuned with
evidence instead of intuition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aka.db.models import ActorType, AuditLog, new_id, utcnow
from aka.obs.logging import get_logger
from aka.obs.telemetry import current_trace_id

log = get_logger("aka.audit")


def record(
    session: Session,
    *,
    tenant_id: str,
    action: str,
    decision: str,
    actor_type: ActorType = ActorType.SYSTEM,
    actor_id: str | None = None,
    subject_type: str = "",
    subject_id: str | None = None,
    rule: str | None = None,
    overrides_audit_id: str | None = None,
    payload: dict | None = None,
) -> AuditLog:
    entry = AuditLog(
        id=new_id(),
        tenant_id=tenant_id,
        actor_type=actor_type,
        actor_id=actor_id,
        action=action,
        subject_type=subject_type,
        subject_id=subject_id,
        decision=decision,
        rule=rule,
        overrides_audit_id=overrides_audit_id,
        payload=payload or {},
        trace_id=current_trace_id(),
    )
    session.add(entry)
    session.flush()
    log.info(
        "audit",
        action=action,
        decision=decision,
        rule=rule,
        actor=actor_type.value,
        subject=subject_id,
    )
    return entry


def latest_system_decision(
    session: Session, *, tenant_id: str, subject_id: str
) -> AuditLog | None:
    """The machine verdict a human review would be overriding."""
    return session.scalars(
        select(AuditLog)
        .where(
            AuditLog.tenant_id == tenant_id,
            AuditLog.subject_id == subject_id,
            AuditLog.actor_type == ActorType.SYSTEM,
        )
        .order_by(AuditLog.created_at.desc())
    ).first()


@dataclass
class OverrideStats:
    total_human_decisions: int
    overrides: int
    window_days: int

    @property
    def rate(self) -> float:
        return self.overrides / self.total_human_decisions if self.total_human_decisions else 0.0

    def as_dict(self) -> dict:
        return {
            "window_days": self.window_days,
            "human_decisions": self.total_human_decisions,
            "overrides": self.overrides,
            "override_rate": round(self.rate, 4),
        }


def override_stats(
    session: Session,
    *,
    tenant_id: str,
    window_days: int = 30,
    rule: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> OverrideStats:
    """Fraction of human decisions that contradicted the automatic verdict.

    A high rate means the thresholds are miscalibrated — either the engine is
    queuing things it should auto-approve, or auto-approving things humans
    reject. Both are actionable; the raw count is not.

    ``since``/``until`` select an explicit period (e.g. before and after a
    threshold change) instead of the trailing ``window_days``.
    """
    since = since or utcnow() - timedelta(days=window_days)

    base = select(func.count(AuditLog.id)).where(
        AuditLog.tenant_id == tenant_id,
        AuditLog.actor_type == ActorType.HUMAN,
        AuditLog.created_at >= since,
    )
    if until is not None:
        base = base.where(AuditLog.created_at < until)
    if rule:
        base = base.where(AuditLog.rule == rule)

    total = session.scalar(base) or 0
    overrides = session.scalar(base.where(AuditLog.overrides_audit_id.is_not(None))) or 0
    return OverrideStats(total_human_decisions=total, overrides=overrides, window_days=window_days)


def rule_breakdown(session: Session, *, tenant_id: str, window_days: int = 30) -> list[dict]:
    """Per-rule decision counts — the table you actually tune thresholds from."""
    since = utcnow() - timedelta(days=window_days)
    rows = session.execute(
        select(
            AuditLog.rule,
            AuditLog.actor_type,
            AuditLog.decision,
            func.count(AuditLog.id),
        )
        .where(AuditLog.tenant_id == tenant_id, AuditLog.created_at >= since)
        .group_by(AuditLog.rule, AuditLog.actor_type, AuditLog.decision)
    ).all()

    grouped: dict[str, dict] = {}
    for rule, actor_type, decision, count in rows:
        key = rule or "unspecified"
        entry = grouped.setdefault(
            key, {"rule": key, "auto": 0, "human": 0, "decisions": {}}
        )
        actor = actor_type.value if hasattr(actor_type, "value") else str(actor_type)
        entry["auto" if actor == "system" else "human"] += count
        entry["decisions"][f"{actor}:{decision}"] = count

    return sorted(grouped.values(), key=lambda e: e["auto"] + e["human"], reverse=True)


def trail_for(session: Session, *, tenant_id: str, subject_id: str) -> list[AuditLog]:
    """Full decision history for one subject, oldest first."""
    return list(
        session.scalars(
            select(AuditLog)
            .where(AuditLog.tenant_id == tenant_id, AuditLog.subject_id == subject_id)
            .order_by(AuditLog.created_at)
        )
    )
