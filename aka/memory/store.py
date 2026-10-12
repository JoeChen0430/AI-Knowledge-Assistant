"""L2 long-term memory store.

Retrieval is not pure similarity. A fact the user stated eight months ago and
has never touched since should lose to one from last week, and a fact the
assistant keeps successfully using should get harder to displace. The ranking
is therefore:

    score = similarity * decay(age) * reinforcement(use_count)

Vectors live as float32 blobs and cosine runs in numpy. That is linear in the
number of live memories, which is the right trade at personal-assistant scale
(hundreds to low thousands of facts per user) and keeps the default deployment
extension-free. The swap to pgvector is confined to :meth:`search`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aka.db.models import MemoryItem, MemoryStatus, MemoryTier, new_id, utcnow
from aka.obs.logging import get_logger
from aka.obs.metrics import memory_items, memory_writes
from aka.settings import Settings, get_settings

log = get_logger("aka.memory.store")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_fact(text: str) -> str:
    """Dedupe key: case-folded, punctuation-free, whitespace-collapsed."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def pack_vector(vec: list[float] | np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes | None, dim: int) -> np.ndarray | None:
    if not blob or dim <= 0:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr if arr.size == dim else None


@dataclass
class MemoryHit:
    item: MemoryItem
    similarity: float
    decay: float
    reinforcement: float

    @property
    def score(self) -> float:
        return self.similarity * self.decay * self.reinforcement

    def as_dict(self) -> dict:
        return {
            "id": self.item.id,
            "content": self.item.content,
            "kind": self.item.kind,
            "confidence": round(self.item.confidence, 3),
            "similarity": round(self.similarity, 4),
            "decay": round(self.decay, 4),
            "score": round(self.score, 4),
        }


def decay_factor(created_at: datetime, half_life_days: float, now: datetime | None = None) -> float:
    """Exponential decay by age. 1.0 when fresh, 0.5 after one half-life."""
    now = now or utcnow()
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def reinforcement_factor(use_count: int) -> float:
    """Mild boost for facts that keep proving useful, saturating so a single
    hot memory cannot dominate every retrieval."""
    return 1.0 + 0.15 * math.log1p(max(0, use_count))


class MemoryStore:
    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()

    # --- reads ------------------------------------------------------------
    def live_items(self, tenant_id: str, user_id: str) -> list[MemoryItem]:
        return list(
            self.session.scalars(
                select(MemoryItem).where(
                    MemoryItem.tenant_id == tenant_id,
                    MemoryItem.user_id == user_id,
                    MemoryItem.tier == MemoryTier.L2,
                    MemoryItem.status == MemoryStatus.ACTIVE,
                )
            )
        )

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query_vector: list[float],
        *,
        k: int | None = None,
        min_score: float | None = None,
    ) -> list[MemoryHit]:
        k = k or self.settings.L2_TOP_K
        min_score = self.settings.L2_MIN_RETRIEVAL_SCORE if min_score is None else min_score

        items = self.live_items(tenant_id, user_id)
        if not items:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            return []
        query = query / qnorm

        now = utcnow()
        hits: list[MemoryHit] = []
        for item in items:
            vec = unpack_vector(item.embedding, item.embedding_dim)
            if vec is None:
                continue
            vnorm = float(np.linalg.norm(vec))
            if vnorm == 0.0:
                continue
            similarity = float(np.dot(query, vec / vnorm))
            if similarity <= 0:
                continue
            hit = MemoryHit(
                item=item,
                similarity=similarity,
                decay=decay_factor(item.created_at, item.half_life_days, now),
                reinforcement=reinforcement_factor(item.use_count),
            )
            if hit.score >= min_score:
                hits.append(hit)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def find_duplicate(self, tenant_id: str, user_id: str, content: str) -> MemoryItem | None:
        """Exact normalized match — the cheap dedupe before the semantic one."""
        return self.session.scalars(
            select(MemoryItem).where(
                MemoryItem.tenant_id == tenant_id,
                MemoryItem.user_id == user_id,
                MemoryItem.normalized == normalize_fact(content),
                MemoryItem.status.in_([MemoryStatus.ACTIVE, MemoryStatus.PENDING]),
            )
        ).first()

    def pending(self, tenant_id: str, user_id: str | None = None) -> list[MemoryItem]:
        q = select(MemoryItem).where(
            MemoryItem.tenant_id == tenant_id, MemoryItem.status == MemoryStatus.PENDING
        )
        if user_id:
            q = q.where(MemoryItem.user_id == user_id)
        return list(self.session.scalars(q.order_by(MemoryItem.created_at.desc())))

    # --- writes -----------------------------------------------------------
    def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        content: str,
        kind: str = "fact",
        confidence: float = 0.5,
        status: MemoryStatus = MemoryStatus.PENDING,
        embedding: list[float] | None = None,
        source_conversation_id: str | None = None,
        source_message_id: str | None = None,
        evidence: str = "",
    ) -> MemoryItem:
        item = MemoryItem(
            id=new_id(),
            tenant_id=tenant_id,
            user_id=user_id,
            tier=MemoryTier.L2,
            kind=kind,
            content=content.strip(),
            normalized=normalize_fact(content),
            confidence=confidence,
            status=status,
            embedding=pack_vector(embedding) if embedding else None,
            embedding_dim=len(embedding) if embedding else 0,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            evidence=evidence[:1000],
            half_life_days=self.settings.L2_HALF_LIFE_DAYS,
        )
        self.session.add(item)
        self.session.flush()
        memory_writes.labels(outcome=status.value).inc()
        return item

    def supersede(self, old: MemoryItem, new: MemoryItem) -> None:
        """Retain the old fact as history rather than deleting it — provenance
        matters when a human later asks why the assistant changed its mind."""
        old.status = MemoryStatus.SUPERSEDED
        old.superseded_by = new.id
        self.session.flush()
        log.info("memory.superseded", old_id=old.id, new_id=new.id)

    def activate(self, item: MemoryItem) -> MemoryItem:
        item.status = MemoryStatus.ACTIVE
        self.session.flush()
        memory_writes.labels(outcome="activated").inc()
        return item

    def reject(self, item: MemoryItem, reason: str = "") -> MemoryItem:
        item.status = MemoryStatus.REJECTED
        item.evidence = f"{item.evidence}\nrejected: {reason}".strip()[:1000]
        self.session.flush()
        memory_writes.labels(outcome="rejected").inc()
        return item

    def mark_used(self, items: list[MemoryItem]) -> None:
        """Reinforcement signal, applied when a memory actually reached the prompt."""
        now = utcnow()
        for item in items:
            item.use_count += 1
            item.last_used_at = now
        self.session.flush()

    def forget(self, item: MemoryItem) -> None:
        self.session.delete(item)
        self.session.flush()

    # --- housekeeping -----------------------------------------------------
    def prune_decayed(self, tenant_id: str, user_id: str, *, floor: float = 0.02) -> int:
        """Drop memories that have decayed below usefulness and were never used.

        A fact that has been reinforced is kept regardless of age — that is the
        difference between decay and a plain TTL.
        """
        removed = 0
        now = utcnow()
        for item in self.live_items(tenant_id, user_id):
            if item.use_count > 0:
                continue
            if decay_factor(item.created_at, item.half_life_days, now) < floor:
                self.session.delete(item)
                removed += 1
        self.session.flush()
        if removed:
            log.info("memory.pruned", tenant=tenant_id, user=user_id, removed=removed)
        return removed

    def refresh_gauge(self, tenant_id: str) -> None:
        count = self.session.scalar(
            select(func.count(MemoryItem.id)).where(
                MemoryItem.tenant_id == tenant_id,
                MemoryItem.status == MemoryStatus.ACTIVE,
            )
        )
        memory_items.labels(tenant=tenant_id).set(count or 0)
