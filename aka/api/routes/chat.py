"""Chat, answer feedback, and conversation history."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aka.api.deps import get_principal, get_services, get_session
from aka.api.schemas import ChatRequest, FeedbackRequest
from aka.api.services import AppServices
from aka.db.models import Conversation, Message
from aka.hitl import approvals, effects
from aka.hitl.policy import Action
from aka.rag.pipeline import OUTCOME_ANSWERED
from aka.security.auth import ROLE_REVIEWER, Principal

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(
    body: ChatRequest,
    background: BackgroundTasks,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    services: AppServices = Depends(get_services),
) -> dict:
    """Answer a question from the caller's documents.

    Not streamed, on purpose: the grounding gate needs the complete answer
    before deciding whether it may be shown at all.
    """
    result = services.pipeline(session).answer(
        body.question,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        conversation_id=body.conversation_id,
    )
    session.commit()

    if services.settings.MEMORY_EXTRACTION_ENABLED and result.outcome == OUTCOME_ANSWERED:
        background.add_task(
            services.learn_in_background,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            conversation_id=result.conversation_id,
            question=body.question,
            answer=result.answer,
        )
    return result.as_dict()


def _owned_message(session: Session, principal: Principal, message_id: str) -> Message:
    message = session.get(Message, message_id)
    if message is None or message.tenant_id != principal.tenant_id or message.role != "assistant":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "message not found")
    convo = session.get(Conversation, message.conversation_id)
    if convo is None or (convo.user_id != principal.user_id and not principal.has_role(ROLE_REVIEWER)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "message not found")
    return message


@router.post("/chat/feedback")
def feedback(
    body: FeedbackRequest,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    services: AppServices = Depends(get_services),
) -> dict:
    """Thumbs up/down on an answer: a human verdict on the release decision.

    Disagreeing with what the gate did (thumbs-down on a released answer,
    thumbs-up on one it queued) counts toward the override rate.
    """
    message = _owned_message(session, principal, body.message_id)
    try:
        resolution = approvals.record_feedback(
            session,
            tenant_id=principal.tenant_id,
            subject_id=message.id,
            approved=body.helpful,
            actor_id=principal.user_id,
            note=body.note,
        )
    except approvals.ApprovalError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    applied = effects.apply_verdict(
        session,
        kind=Action.ANSWER_RELEASE.value,
        tenant_id=principal.tenant_id,
        subject_id=message.id,
        approved=body.helpful,
        settings=services.settings,
    )
    session.commit()
    return {
        "recorded": True,
        "was_override": resolution.was_override,
        "system_verdict": resolution.system_verdict,
        "effect": applied,
    }


@router.get("/conversations")
def list_conversations(
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
) -> list[dict]:
    last_activity = (
        select(Message.conversation_id, func.max(Message.created_at).label("last_at"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = session.execute(
        select(Conversation, last_activity.c.last_at)
        .join(last_activity, last_activity.c.conversation_id == Conversation.id)
        .where(
            Conversation.tenant_id == principal.tenant_id,
            Conversation.user_id == principal.user_id,
            Conversation.is_archived.is_(False),
        )
        .order_by(last_activity.c.last_at.desc())
        .limit(100)
    ).all()
    return [
        {"id": convo.id, "title": convo.title, "last_activity": last_at.isoformat() if last_at else None}
        for convo, last_at in rows
    ]


def _owned_conversation(session: Session, principal: Principal, conversation_id: str) -> Conversation:
    convo = session.get(Conversation, conversation_id)
    if (
        convo is None
        or convo.tenant_id != principal.tenant_id
        or convo.user_id != principal.user_id
        or convo.is_archived
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return convo


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
) -> dict:
    convo = _owned_conversation(session, principal, conversation_id)
    messages = session.scalars(
        select(Message).where(Message.conversation_id == convo.id).order_by(Message.seq)
    )
    return {
        "id": convo.id,
        "title": convo.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "meta": m.meta or {},
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def archive_conversation(
    conversation_id: str,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
) -> None:
    """Archived, not deleted: the audit trail references its messages."""
    convo = _owned_conversation(session, principal, conversation_id)
    convo.is_archived = True
    session.commit()
