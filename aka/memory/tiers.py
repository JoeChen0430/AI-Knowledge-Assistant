"""Tiered memory orchestration.

    L0  working set   last N turns, verbatim, bounded by a token budget
    L1  session       rolling summary of turns that fell out of L0
    L2  long-term     durable facts, decayed, retrieved by relevance
    L3  corpus        the documents (owned by aka/rag, not here)

Reading is cheap and always happens. Writing to L2 is the expensive, risky
direction — it is what makes the assistant "remember" something wrong forever —
so promotion runs through :func:`decide_promotion` and lands in one of three
places: committed, queued for a human, or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aka.db.models import (
    Conversation,
    Message,
    MemoryItem,
    MemoryStatus,
    SessionSummary,
    new_id,
    utcnow,
)
from aka.ingest.hashing import estimate_tokens
from aka.llm.client import ChatMessage
from aka.memory.extraction import FactCandidate, detect_contradiction, summarize_turns
from aka.memory.store import MemoryHit, MemoryStore
from aka.obs.logging import get_logger
from aka.obs.metrics import memory_promotions
from aka.obs.telemetry import stage
from aka.settings import Settings, get_settings

log = get_logger("aka.memory.tiers")


# --------------------------------------------------------------------------
# Promotion policy
# --------------------------------------------------------------------------

COMMIT = "commit"
REVIEW = "review"
DROP = "drop"


@dataclass
class PromotionDecision:
    action: str            # commit | review | drop
    reason: str
    confidence: float
    conflicts_with: MemoryItem | None = None

    @property
    def status(self) -> MemoryStatus:
        return {
            COMMIT: MemoryStatus.ACTIVE,
            REVIEW: MemoryStatus.PENDING,
            DROP: MemoryStatus.REJECTED,
        }[self.action]


def decide_promotion(
    candidate: FactCandidate,
    *,
    settings: Settings,
    duplicate: MemoryItem | None = None,
    conflict: MemoryItem | None = None,
) -> PromotionDecision:
    """Pure function — the entire L1→L2 gate, so it is trivially testable.

    A contradiction always goes to a human regardless of confidence: silently
    overwriting something the user previously said is the failure mode this
    whole mechanism exists to prevent.
    """
    if duplicate is not None:
        return PromotionDecision(DROP, "already stored", candidate.confidence)

    if candidate.confidence < settings.MEMORY_REJECT_CONFIDENCE:
        return PromotionDecision(DROP, "confidence below floor", candidate.confidence)

    if conflict is not None:
        return PromotionDecision(
            REVIEW, "contradicts an existing memory", candidate.confidence, conflict
        )

    if candidate.confidence >= settings.MEMORY_AUTO_COMMIT_CONFIDENCE:
        return PromotionDecision(COMMIT, "high confidence, no conflict", candidate.confidence)

    return PromotionDecision(REVIEW, "confidence below auto-commit threshold", candidate.confidence)


# --------------------------------------------------------------------------
# Assembled context
# --------------------------------------------------------------------------

@dataclass
class MemoryContext:
    """Everything the prompt builder needs from memory, plus the accounting
    that lets the trace explain where the token budget went."""

    recent_turns: list[ChatMessage] = field(default_factory=list)
    session_summary: str = ""
    facts: list[MemoryHit] = field(default_factory=list)
    l0_tokens: int = 0
    l1_tokens: int = 0
    l2_tokens: int = 0
    turns_condensed: int = 0

    @property
    def total_tokens(self) -> int:
        return self.l0_tokens + self.l1_tokens + self.l2_tokens

    def as_dict(self) -> dict:
        return {
            "recent_turns": len(self.recent_turns),
            "has_summary": bool(self.session_summary),
            "facts": [f.as_dict() for f in self.facts],
            "tokens": {
                "l0": self.l0_tokens,
                "l1": self.l1_tokens,
                "l2": self.l2_tokens,
                "total": self.total_tokens,
            },
            "turns_condensed": self.turns_condensed,
        }


class TieredMemory:
    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.store = MemoryStore(session, self.settings)

    # --- L0 ---------------------------------------------------------------
    def recent_messages(self, conversation_id: str) -> list[Message]:
        """Last N turns, then trimmed to the token budget from the newest back.

        Trimming from the newest end matters: the most recent turn is the one
        the user's question actually refers to.
        """
        rows = list(
            self.session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.seq.desc())
                .limit(self.settings.L0_MAX_TURNS * 2)
            )
        )
        rows.reverse()

        budget = self.settings.L0_TOKEN_BUDGET
        kept: list[Message] = []
        used = 0
        for msg in reversed(rows):
            cost = estimate_tokens(msg.content)
            if used + cost > budget and kept:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        return kept

    # --- L1 ---------------------------------------------------------------
    def current_summary(self, conversation_id: str) -> SessionSummary | None:
        return self.session.scalars(
            select(SessionSummary)
            .where(SessionSummary.conversation_id == conversation_id)
            .order_by(SessionSummary.covers_through_seq.desc())
        ).first()

    def condense_if_needed(self, conversation_id: str, router, *, deadline=None) -> SessionSummary | None:
        """Fold anything older than the L0 window into the rolling summary."""
        kept = self.recent_messages(conversation_id)
        oldest_kept_seq = kept[0].seq if kept else 0

        existing = self.current_summary(conversation_id)
        covered_through = existing.covers_through_seq if existing else -1

        stale = list(
            self.session.scalars(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.seq < oldest_kept_seq,
                    Message.seq > covered_through,
                )
                .order_by(Message.seq)
            )
        )
        if not stale:
            return existing

        transcript = "\n".join(f"{m.role}: {m.content}" for m in stale)
        with stage("memory.l1_condense", **{"aka.messages_condensed": len(stale)}):
            summary_text = summarize_turns(
                router,
                transcript=transcript,
                previous_summary=existing.summary if existing else "",
                deadline=deadline,
            )

        if not summary_text.strip():
            return existing

        summary = SessionSummary(
            id=new_id(),
            conversation_id=conversation_id,
            tenant_id=stale[0].tenant_id,
            summary=summary_text,
            covers_through_seq=stale[-1].seq,
            token_count=estimate_tokens(summary_text),
            expires_at=utcnow() + timedelta(seconds=self.settings.L1_TTL_S),
        )
        self.session.add(summary)
        self.session.flush()
        log.info(
            "memory.l1_condensed",
            conversation=conversation_id,
            messages=len(stale),
            summary_tokens=summary.token_count,
        )
        return summary

    # --- assembly ---------------------------------------------------------
    def build_context(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        query_vector: list[float] | None = None,
    ) -> MemoryContext:
        ctx = MemoryContext()

        with stage("memory.assemble"):
            messages = self.recent_messages(conversation_id)
            ctx.recent_turns = [
                ChatMessage(role=m.role, content=m.content) for m in messages
            ]
            ctx.l0_tokens = sum(estimate_tokens(m.content) for m in messages)

            summary = self.current_summary(conversation_id)
            if summary is not None:
                ctx.session_summary = summary.summary
                ctx.l1_tokens = summary.token_count
                ctx.turns_condensed = summary.covers_through_seq + 1

            if query_vector:
                ctx.facts = self.store.search(tenant_id, user_id, query_vector)
                ctx.l2_tokens = sum(estimate_tokens(f.item.content) for f in ctx.facts)
                if ctx.facts:
                    self.store.mark_used([f.item for f in ctx.facts])

        return ctx

    # --- promotion --------------------------------------------------------
    def promote(
        self,
        candidate: FactCandidate,
        *,
        tenant_id: str,
        user_id: str,
        router,
        embedding: list[float] | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        deadline: float | None = None,
    ) -> tuple[MemoryItem | None, PromotionDecision]:
        """Run one candidate through the gate and persist the outcome."""
        duplicate = self.store.find_duplicate(tenant_id, user_id, candidate.content)

        conflict = None
        if duplicate is None and embedding:
            # Only the nearest existing memory is worth a contradiction check;
            # checking all of them would cost one LLM call per stored fact.
            neighbours = self.store.search(
                tenant_id, user_id, embedding, k=1, min_score=0.5
            )
            if neighbours:
                verdict = detect_contradiction(
                    router,
                    new_fact=candidate.content,
                    existing_fact=neighbours[0].item.content,
                    deadline=deadline,
                )
                if verdict.contradicts and verdict.confidence >= 0.5:
                    conflict = neighbours[0].item

        decision = decide_promotion(
            candidate, settings=self.settings, duplicate=duplicate, conflict=conflict
        )
        memory_promotions.labels(decision=decision.action).inc()

        if decision.action == DROP:
            log.debug("memory.promotion_dropped", reason=decision.reason)
            return None, decision

        item = self.store.create(
            tenant_id=tenant_id,
            user_id=user_id,
            content=candidate.content,
            kind=candidate.kind,
            confidence=candidate.confidence,
            status=decision.status,
            embedding=embedding,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            evidence=candidate.evidence,
        )

        if decision.action == COMMIT and conflict is not None:
            self.store.supersede(conflict, item)

        log.info(
            "memory.promotion",
            action=decision.action,
            reason=decision.reason,
            confidence=round(candidate.confidence, 3),
            memory_id=item.id,
        )
        return item, decision

    # --- persistence helpers ---------------------------------------------
    def append_message(
        self,
        *,
        conversation_id: str,
        tenant_id: str,
        role: str,
        content: str,
        trace_id: str | None = None,
        meta: dict | None = None,
        message_id: str | None = None,
    ) -> Message:
        # `or -1` would be wrong here: a max seq of 0 is falsy, which would
        # hand out seq 0 forever and destroy message ordering.
        highest = self.session.scalar(
            select(func.max(Message.seq)).where(Message.conversation_id == conversation_id)
        )
        next_seq = 0 if highest is None else highest + 1

        message = Message(
            id=message_id or new_id(),
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            trace_id=trace_id,
            seq=next_seq,
            meta=meta or {},
        )
        self.session.add(message)
        self.session.flush()
        return message

    def ensure_conversation(
        self, *, conversation_id: str | None, tenant_id: str, user_id: str, title: str = ""
    ) -> Conversation:
        claimed_id = conversation_id
        if conversation_id:
            existing = self.session.get(Conversation, conversation_id)
            # Scope check: a conversation id from the client must belong to the
            # authenticated tenant, or it is someone else's history.
            if existing is not None:
                if existing.tenant_id == tenant_id:
                    return existing
                # Cross-tenant id: mint a fresh one. Reusing the requested id
                # would collide with the real owner's row.
                log.warning(
                    "memory.conversation_scope_violation",
                    requested=conversation_id,
                    tenant=tenant_id,
                )
                claimed_id = None

        convo = Conversation(
            id=claimed_id or new_id(),
            tenant_id=tenant_id,
            user_id=user_id,
            title=(title or "New Chat")[:255],
        )
        self.session.add(convo)
        self.session.flush()
        return convo
