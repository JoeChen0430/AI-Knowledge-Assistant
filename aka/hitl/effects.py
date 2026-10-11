"""Carrying out a human verdict.

The approval queue and the audit log record *that* a human decided; this
module makes the decision take effect on the thing it was about. Kept apart so
the queue stays a pure record and every effect is one small, testable function.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from aka.db.models import MemoryItem, MemoryStatus, Message
from aka.hitl.policy import Action
from aka.memory.store import MemoryStore
from aka.obs.logging import get_logger
from aka.settings import Settings, get_settings

log = get_logger("aka.hitl.effects")


def apply_verdict(
    session: Session,
    *,
    kind: str,
    tenant_id: str,
    subject_id: str | None,
    approved: bool,
    context: dict | None = None,
    settings: Settings | None = None,
) -> dict:
    """Apply an approval or rejection. Returns a summary for the API response."""
    settings = settings or get_settings()
    context = context or {}
    if not subject_id:
        return {"applied": False, "reason": "verdict has no subject"}

    if kind == Action.MEMORY_WRITE.value:
        return _memory_write(session, tenant_id, subject_id, approved, context, settings)
    if kind == Action.ANSWER_RELEASE.value:
        return _answer_release(session, tenant_id, subject_id, approved)
    return {"applied": False, "reason": f"no effect defined for '{kind}'"}


def _memory_write(
    session: Session,
    tenant_id: str,
    subject_id: str,
    approved: bool,
    context: dict,
    settings: Settings,
) -> dict:
    item = session.get(MemoryItem, subject_id)
    if item is None or item.tenant_id != tenant_id:
        return {"applied": False, "reason": "memory item not found"}

    store = MemoryStore(session, settings)
    if approved:
        if item.status is MemoryStatus.PENDING:
            store.activate(item)
        # An approved contradiction means the new fact replaces the old one.
        conflict_id = (context.get("review_context") or {}).get("conflicts_with")
        if conflict_id:
            old = session.get(MemoryItem, conflict_id)
            if old is not None and old.tenant_id == tenant_id and old.status is MemoryStatus.ACTIVE:
                store.supersede(old, item)
    elif item.status in (MemoryStatus.PENDING, MemoryStatus.ACTIVE):
        # Covers both a queued write and an auto-committed memory the user
        # asked to forget: either way it must stop reaching prompts.
        store.reject(item, reason="rejected by a human")

    store.refresh_gauge(tenant_id)
    log.info("hitl.memory_verdict_applied", memory_id=item.id, status=item.status.value)
    return {"applied": True, "memory_status": item.status.value}


def _answer_release(session: Session, tenant_id: str, subject_id: str, approved: bool) -> dict:
    message = session.get(Message, subject_id)
    if message is None or message.tenant_id != tenant_id:
        return {"applied": False, "reason": "message not found"}
    # Reassign rather than mutate: plain JSON columns do not track in-place edits.
    message.meta = {**(message.meta or {}), "review": "approved" if approved else "rejected"}
    session.flush()
    return {"applied": True, "review": message.meta["review"]}
