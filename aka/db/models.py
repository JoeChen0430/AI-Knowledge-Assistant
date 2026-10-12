"""Relational schema.

Runs on SQLite (laptop, zero setup) or Postgres (compose) through the same
SQLAlchemy models. Vectors are stored as raw float32 blobs rather than pgvector
so the default path needs no extension; the L2 store does cosine in numpy,
which is fine into the 10^5-row range. Past that, swap MemoryItem.embedding for
a pgvector column — the query interface in aka/memory/store.py is the only
thing that would change.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend.

    SQLite has no timezone type: SQLAlchemy stores the wall-clock value and
    hands back a naive datetime, which then raises on comparison with
    ``utcnow()`` (e.g. an approval's expiry check). Normalise to UTC on the
    way in and re-attach UTC on the way out, so Python always sees aware
    values regardless of the database.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utcnow, onupdate=utcnow
    )


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class DocumentStatus(str, enum.Enum):
    PENDING = "pending"          # uploaded, not yet indexed
    INDEXED = "indexed"
    QUARANTINED = "quarantined"  # secret/PII hit — awaits human approval
    DELETED = "deleted"          # soft delete, recoverable within UNDO_WINDOW
    FAILED = "failed"


class IndexStatus(str, enum.Enum):
    BUILDING = "building"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"    # will retry
    DEAD = "dead"        # exhausted attempts -> dead-letter queue


class MemoryTier(str, enum.Enum):
    L1 = "l1"  # rolling session summary
    L2 = "l2"  # durable extracted fact


class MemoryStatus(str, enum.Enum):
    ACTIVE = "active"
    PENDING = "pending"        # below auto-commit confidence -> approval queue
    REJECTED = "rejected"
    SUPERSEDED = "superseded"  # replaced by a newer, contradicting fact


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"   # policy allowed it without a human
    AUTO_REJECTED = "auto_rejected"


class ActorType(str, enum.Enum):
    SYSTEM = "system"
    HUMAN = "human"


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    email: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str | None] = mapped_column(String(255), default=None)
    role: Mapped[str] = mapped_column(String(32), default="user")  # user | reviewer | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_user_tenant_email"),)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    rel_path: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str | None] = mapped_column(String(128), default=None)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False), default=DocumentStatus.PENDING, index=True
    )
    quarantine_reason: Mapped[str | None] = mapped_column(Text, default=None)
    trash_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    uploaded_by: Mapped[str | None] = mapped_column(String(32), default=None)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # One live row per (tenant, path); re-uploads update the hash in place.
        UniqueConstraint("tenant_id", "rel_path", name="uq_document_tenant_path"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
    )


class Chunk(Base):
    """One embedded unit. ``chunk_hash`` is what makes ingest incremental:
    a chunk whose hash already exists in the active index is never re-embedded.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # deterministic
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    text_preview: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunks_tenant_hash", "tenant_id", "chunk_hash"),)


class IndexVersion(Base, TimestampMixin):
    """Blue/green pointer. Readers resolve the ACTIVE row; a build writes into a
    new collection and only flips status once smoke checks pass.
    """

    __tablename__ = "index_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    collection_name: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[IndexStatus] = mapped_column(
        Enum(IndexStatus, native_enum=False), default=IndexStatus.BUILDING, index=True
    )
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embed_model: Mapped[str] = mapped_column(String(128), default="")
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    __table_args__ = (Index("ix_index_versions_tenant_status", "tenant_id", "status"),)


# --------------------------------------------------------------------------
# Durable job queue
# --------------------------------------------------------------------------

class Job(Base, TimestampMixin):
    """DB-backed queue: at-least-once delivery with leases, capped retries with
    exponential backoff, and a dead-letter state. Chosen over Redis/Celery so
    the default deployment needs no broker; the lease pattern is the same.
    """

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), default=JobStatus.QUEUED, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    # Idempotency key collapses duplicate submissions (e.g. double-clicked upload).
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, default=None)
    run_after: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    locked_by: Mapped[str | None] = mapped_column(String(64), default=None)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(32), default=None)
    result: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_jobs_claim", "status", "run_after"),)


# --------------------------------------------------------------------------
# Conversations and tiered memory
# --------------------------------------------------------------------------

class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New Chat")
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_convo_seq", "conversation_id", "seq"),)


class SessionSummary(Base):
    """L1: rolling condensation of turns that fell out of the L0 window."""

    __tablename__ = "session_summaries"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    covers_through_seq: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)


class MemoryItem(Base, TimestampMixin):
    """L2: a durable fact about the user, with provenance and a decay clock."""

    __tablename__ = "memory_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    tier: Mapped[MemoryTier] = mapped_column(
        Enum(MemoryTier, native_enum=False), default=MemoryTier.L2
    )
    kind: Mapped[str] = mapped_column(String(32), default="fact")  # fact | preference | identity
    content: Mapped[str] = mapped_column(Text)
    normalized: Mapped[str] = mapped_column(Text, default="")  # dedupe key
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0)

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[MemoryStatus] = mapped_column(
        Enum(MemoryStatus, native_enum=False), default=MemoryStatus.PENDING, index=True
    )
    superseded_by: Mapped[str | None] = mapped_column(String(32), default=None)

    # Provenance — every fact must point back at the turn that produced it.
    source_conversation_id: Mapped[str | None] = mapped_column(String(32), default=None)
    source_message_id: Mapped[str | None] = mapped_column(String(32), default=None)
    evidence: Mapped[str] = mapped_column(Text, default="")

    # Decay/reinforcement signals.
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    half_life_days: Mapped[float] = mapped_column(Float, default=30.0)

    __table_args__ = (
        Index("ix_memory_tenant_user_status", "tenant_id", "user_id", "status"),
        Index("ix_memory_norm", "tenant_id", "user_id", "normalized"),
    )


# --------------------------------------------------------------------------
# Human-in-the-loop
# --------------------------------------------------------------------------

class Approval(Base, TimestampMixin):
    """One queued decision awaiting a human. ``kind`` maps to a policy rule."""

    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    subject_type: Mapped[str] = mapped_column(String(64))
    subject_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    rule: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[ApprovalStatus] = mapped_column(
        Enum(ApprovalStatus, native_enum=False), default=ApprovalStatus.PENDING, index=True
    )
    requested_by: Mapped[str | None] = mapped_column(String(32), default=None)
    resolved_by: Mapped[str | None] = mapped_column(String(32), default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    resolution_note: Mapped[str | None] = mapped_column(Text, default=None)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    trace_id: Mapped[str | None] = mapped_column(String(32), default=None)

    __table_args__ = (Index("ix_approvals_tenant_status_kind", "tenant_id", "status", "kind"),)


class AuditLog(Base):
    """Append-only record of every consequential decision, automatic or human.

    This is what makes the override rate measurable: automatic verdicts and the
    human verdicts that later contradict them land in the same table.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    actor_type: Mapped[ActorType] = mapped_column(Enum(ActorType, native_enum=False))
    actor_id: Mapped[str | None] = mapped_column(String(64), default=None)
    action: Mapped[str] = mapped_column(String(64), index=True)
    subject_type: Mapped[str] = mapped_column(String(64), default="")
    subject_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    decision: Mapped[str] = mapped_column(String(32), default="")
    rule: Mapped[str | None] = mapped_column(String(64), default=None)
    overrides_audit_id: Mapped[str | None] = mapped_column(String(32), default=None)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)

    __table_args__ = (Index("ix_audit_tenant_action_time", "tenant_id", "action", "created_at"),)


class UndoToken(Base):
    """Reversal handle for a destructive action, valid for a bounded window."""

    __tablename__ = "undo_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str | None] = mapped_column(String(32), default=None)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
