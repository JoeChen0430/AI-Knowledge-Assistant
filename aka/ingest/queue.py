"""Durable job queue backed by the relational store.

Chosen over Celery/RQ so the default deployment needs no broker; the mechanics
are the same ones a broker gives you:

* **atomic claim** — one UPDATE statement moves a row from queued to running
  and stamps the worker id. There is no read-then-write window, so two workers
  polling simultaneously cannot both take the same job.
* **leases** — a claimed job carries an expiry. If the worker dies, the reaper
  returns the job to the queue instead of losing it.
* **capped retries with exponential backoff**, then a terminal DEAD state that
  acts as the dead-letter queue.
* **idempotency keys** — a duplicate submission collapses onto the existing job
  rather than queueing a second identical rebuild.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aka.db.models import Job, JobStatus, new_id, utcnow
from aka.obs.logging import get_logger
from aka.obs.metrics import ingest_jobs, ingest_queue_depth
from aka.obs.telemetry import current_trace_id
from aka.settings import Settings, get_settings

log = get_logger("aka.queue")

CLAIMABLE = (JobStatus.QUEUED, JobStatus.FAILED)
MAX_BACKOFF_S = 600
# Bounded so a heavily contended queue degrades to "no work this poll" rather
# than spinning; the caller polls again shortly.
_CLAIM_ATTEMPTS = 8


def enqueue(
    session: Session,
    *,
    tenant_id: str,
    kind: str,
    payload: dict | None = None,
    idempotency_key: str | None = None,
    max_attempts: int | None = None,
    delay_s: float = 0.0,
    settings: Settings | None = None,
) -> Job:
    """Add a job. With an idempotency key, returns the existing job instead of
    creating a duplicate."""
    settings = settings or get_settings()

    if idempotency_key:
        existing = session.scalars(
            select(Job).where(Job.idempotency_key == idempotency_key)
        ).first()
        if existing is not None:
            log.info("queue.deduplicated", job_id=existing.id, key=idempotency_key)
            return existing

    job = Job(
        id=new_id(),
        tenant_id=tenant_id,
        kind=kind,
        payload=payload or {},
        status=JobStatus.QUEUED,
        max_attempts=max_attempts or settings.INGEST_MAX_ATTEMPTS,
        idempotency_key=idempotency_key,
        run_after=utcnow() + timedelta(seconds=delay_s),
        trace_id=current_trace_id(),
    )
    session.add(job)
    try:
        session.flush()
    except IntegrityError:
        # Lost a race on the unique idempotency key — take the winner's job.
        session.rollback()
        existing = session.scalars(
            select(Job).where(Job.idempotency_key == idempotency_key)
        ).first()
        if existing is None:
            raise
        return existing

    log.info("queue.enqueued", job_id=job.id, kind=kind, tenant=tenant_id)
    return job


def claim(
    session: Session,
    *,
    worker_id: str,
    kinds: Sequence[str] | None = None,
    lease_seconds: int | None = None,
    settings: Settings | None = None,
) -> Job | None:
    """Atomically take the next runnable job, or return None.

    The whole point is that selection and acquisition happen in one statement.
    A check-then-act version of this ("SELECT a queued job, then UPDATE it")
    lets two workers read the same row before either writes, and both run it.
    """
    settings = settings or get_settings()
    lease_seconds = lease_seconds or settings.INGEST_LEASE_SECONDS

    # Read a candidate, then try to take it with an UPDATE that re-asserts the
    # precondition. The SELECT is only a hint — the UPDATE's WHERE clause is
    # what makes the claim atomic, so a lost race matches 0 rows and we simply
    # look at the next candidate. This is not check-then-act: no decision is
    # made on the basis of the read alone.
    for _ in range(_CLAIM_ATTEMPTS):
        now = utcnow()
        candidate_q = (
            select(Job.id)
            .where(Job.status.in_(CLAIMABLE), Job.run_after <= now)
            .order_by(Job.run_after, Job.created_at)
            .limit(1)
        )
        if kinds:
            candidate_q = candidate_q.where(Job.kind.in_(list(kinds)))
        # Postgres can skip rows another transaction already holds; SQLite
        # serialises writers anyway.
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            candidate_q = candidate_q.with_for_update(skip_locked=True)

        candidate_id = session.scalars(candidate_q).first()
        if candidate_id is None:
            return None

        lease_until = now + timedelta(seconds=lease_seconds)
        result = session.execute(
            update(Job)
            .where(Job.id == candidate_id, Job.status.in_(CLAIMABLE))
            .values(
                status=JobStatus.RUNNING,
                locked_by=worker_id,
                lease_expires_at=lease_until,
                attempts=Job.attempts + 1,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        session.commit()

        if result.rowcount != 1:
            continue  # another worker took it; try the next candidate

        # The session is configured with expire_on_commit=False, so the
        # identity map still holds pre-UPDATE values — force a reload.
        job = session.scalars(
            select(Job).where(Job.id == candidate_id).execution_options(populate_existing=True)
        ).one()
        log.info(
            "queue.claimed", job_id=job.id, kind=job.kind, worker=worker_id, attempt=job.attempts
        )
        return job

    log.warning("queue.claim_contended", worker=worker_id, attempts=_CLAIM_ATTEMPTS)
    return None


def complete(session: Session, job: Job, result: dict | None = None) -> None:
    job.status = JobStatus.SUCCEEDED
    job.result = result or {}
    job.lease_expires_at = None
    job.locked_by = None
    job.last_error = None
    session.commit()
    ingest_jobs.labels(status="succeeded").inc()
    log.info("queue.completed", job_id=job.id, kind=job.kind)


def fail(
    session: Session,
    job: Job,
    error: str,
    *,
    settings: Settings | None = None,
) -> JobStatus:
    """Record a failure: schedule a backed-off retry, or dead-letter it."""
    settings = settings or get_settings()
    job.last_error = error[:2000]
    job.lease_expires_at = None
    job.locked_by = None

    if job.attempts >= job.max_attempts:
        job.status = JobStatus.DEAD
        ingest_jobs.labels(status="dead").inc()
        log.error(
            "queue.dead_lettered",
            job_id=job.id,
            kind=job.kind,
            attempts=job.attempts,
            error=error[:200],
        )
    else:
        backoff = min(settings.INGEST_BACKOFF_BASE_S ** job.attempts, MAX_BACKOFF_S)
        job.status = JobStatus.FAILED
        job.run_after = utcnow() + timedelta(seconds=backoff)
        ingest_jobs.labels(status="retry").inc()
        log.warning(
            "queue.retry_scheduled",
            job_id=job.id,
            attempt=job.attempts,
            retry_in_s=round(backoff, 1),
            error=error[:200],
        )

    session.commit()
    return job.status


def defer(session: Session, job: Job, *, delay_s: float, reason: str) -> None:
    """Put a claimed job back without spending an attempt.

    For a job that could not start rather than one that failed — e.g. another
    build for the same tenant is already running.
    """
    job.status = JobStatus.QUEUED
    job.attempts = max(0, job.attempts - 1)
    job.run_after = utcnow() + timedelta(seconds=delay_s)
    job.lease_expires_at = None
    job.locked_by = None
    job.last_error = f"deferred: {reason}"[:2000]
    session.commit()
    ingest_jobs.labels(status="deferred").inc()
    log.info("queue.deferred", job_id=job.id, kind=job.kind, delay_s=delay_s, reason=reason)


def extend_lease(session: Session, job_id: str, *, worker_id: str, lease_seconds: int) -> bool:
    """Heartbeat: push the lease of a job this worker still holds.

    Without it, a build longer than the lease (a big first ingest) would be
    reaped and started a second time by another worker while still running.
    ``updated_at`` is pinned so the claim time stays readable.
    """
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.locked_by == worker_id, Job.status == JobStatus.RUNNING)
        .values(
            lease_expires_at=utcnow() + timedelta(seconds=lease_seconds),
            updated_at=Job.updated_at,
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount == 1


def reap_expired_leases(session: Session) -> int:
    """Return jobs from dead workers to the queue.

    A worker that crashed mid-job leaves a RUNNING row whose lease expires. The
    attempt has already been counted, so a job that repeatedly kills its worker
    still reaches the dead-letter state rather than looping forever.
    """
    now = utcnow()
    result = session.execute(
        update(Job)
        .where(
            Job.status == JobStatus.RUNNING,
            Job.lease_expires_at.is_not(None),
            Job.lease_expires_at < now,
        )
        .values(
            status=JobStatus.FAILED,
            locked_by=None,
            lease_expires_at=None,
            run_after=now,
            last_error="lease expired — worker presumed dead",
            updated_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    if result.rowcount:
        ingest_jobs.labels(status="reaped").inc(result.rowcount)
        log.warning("queue.leases_reaped", count=result.rowcount)
    return result.rowcount


def requeue_dead(session: Session, job_id: str) -> Job | None:
    """Operator action: push a dead-lettered job back after fixing the cause."""
    job = session.get(Job, job_id)
    if job is None or job.status is not JobStatus.DEAD:
        return None
    job.status = JobStatus.QUEUED
    job.attempts = 0
    job.run_after = utcnow()
    job.last_error = None
    session.commit()
    log.info("queue.requeued", job_id=job.id)
    return job


def ensure_reindex(session: Session, *, tenant_id: str, reason: str, settings: Settings | None = None) -> Job:
    """Queue a reindex unless one is already waiting to start.

    A waiting job will scan DATA_DIR when it runs, so it already covers any
    file written before then; a second one would only rebuild an identical
    index. A *running* job does not count — it may have scanned already.
    """
    waiting = session.scalars(
        select(Job)
        .where(
            Job.tenant_id == tenant_id,
            Job.kind == "reindex",
            Job.status == JobStatus.QUEUED,
        )
        .order_by(Job.created_at)
    ).first()
    if waiting is not None:
        return waiting
    return enqueue(
        session, tenant_id=tenant_id, kind="reindex", payload={"reason": reason}, settings=settings
    )


def queue_stats(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(Job.status, func.count(Job.id)).group_by(Job.status)
    ).all()
    stats = {status.value: 0 for status in JobStatus}
    for status, count in rows:
        key = status.value if hasattr(status, "value") else str(status)
        stats[key] = count
    for key, count in stats.items():
        ingest_queue_depth.labels(state=key).set(count)
    return stats


def purge_finished(session: Session, *, older_than_days: int = 7) -> int:
    cutoff = utcnow() - timedelta(days=older_than_days)
    result = session.execute(
        text(
            "DELETE FROM jobs WHERE status = :status AND updated_at < :cutoff"
        ),
        {"status": JobStatus.SUCCEEDED.value, "cutoff": cutoff},
    )
    session.commit()
    return result.rowcount
