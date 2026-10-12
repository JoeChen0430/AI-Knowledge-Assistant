"""Job queue semantics: atomic claim, retries, dead-lettering, lease recovery."""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from sqlalchemy import select

from aka.db.models import Job, JobStatus, utcnow
from aka.ingest import queue


def test_enqueue_and_claim(session, settings):
    job = queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()

    claimed = queue.claim(session, worker_id="w1", settings=settings)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status is JobStatus.RUNNING
    assert claimed.locked_by == "w1"
    assert claimed.attempts == 1
    assert claimed.lease_expires_at is not None


def test_claim_returns_none_when_empty(session, settings):
    assert queue.claim(session, worker_id="w1", settings=settings) is None


def test_claim_respects_run_after(session, settings):
    job = queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    job.run_after = utcnow() + timedelta(minutes=5)
    session.commit()
    assert queue.claim(session, worker_id="w1", settings=settings) is None


def test_claim_filters_by_kind(session, settings):
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    assert queue.claim(session, worker_id="w1", kinds=["prune_trash"], settings=settings) is None
    assert queue.claim(session, worker_id="w1", kinds=["reindex"], settings=settings) is not None


def test_idempotency_key_collapses_duplicates(session, settings):
    a = queue.enqueue(
        session, tenant_id="t1", kind="reindex", idempotency_key="rebuild:t1", settings=settings
    )
    b = queue.enqueue(
        session, tenant_id="t1", kind="reindex", idempotency_key="rebuild:t1", settings=settings
    )
    session.commit()
    assert a.id == b.id
    assert session.scalar(select(Job).where(Job.id == a.id)) is not None
    assert len(list(session.scalars(select(Job)))) == 1


def test_failure_schedules_backed_off_retry(session, settings):
    settings.INGEST_MAX_ATTEMPTS = 3
    settings.INGEST_BACKOFF_BASE_S = 2.0
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()

    job = queue.claim(session, worker_id="w1", settings=settings)
    status = queue.fail(session, job, "boom", settings=settings)

    assert status is JobStatus.FAILED
    assert job.run_after > utcnow(), "retry must be scheduled in the future"
    assert job.locked_by is None


def test_backoff_grows_with_attempts(session, settings):
    settings.INGEST_MAX_ATTEMPTS = 5
    settings.INGEST_BACKOFF_BASE_S = 2.0
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()

    delays = []
    for _ in range(3):
        job = session.scalars(select(Job)).one()
        job.run_after = utcnow() - timedelta(seconds=1)
        session.commit()
        claimed = queue.claim(session, worker_id="w1", settings=settings)
        before = utcnow()
        queue.fail(session, claimed, "boom", settings=settings)
        delays.append((claimed.run_after - before).total_seconds())

    assert delays[0] < delays[1] < delays[2], f"expected growing backoff, got {delays}"


def test_exhausted_attempts_dead_letter(session, settings):
    settings.INGEST_MAX_ATTEMPTS = 2
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()

    for _ in range(2):
        job = session.scalars(select(Job)).one()
        job.run_after = utcnow() - timedelta(seconds=1)
        session.commit()
        claimed = queue.claim(session, worker_id="w1", settings=settings)
        status = queue.fail(session, claimed, "boom", settings=settings)

    assert status is JobStatus.DEAD
    assert queue.claim(session, worker_id="w1", settings=settings) is None, (
        "dead jobs must not be redelivered"
    )


def test_requeue_dead_job(session, settings):
    settings.INGEST_MAX_ATTEMPTS = 1
    job = queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    claimed = queue.claim(session, worker_id="w1", settings=settings)
    queue.fail(session, claimed, "boom", settings=settings)

    revived = queue.requeue_dead(session, job.id)
    assert revived is not None
    assert revived.status is JobStatus.QUEUED
    assert revived.attempts == 0
    assert queue.claim(session, worker_id="w2", settings=settings) is not None


def test_expired_lease_is_reaped(session, settings):
    """Simulates a worker crashing mid-job."""
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    claimed = queue.claim(session, worker_id="dead-worker", settings=settings)

    claimed.lease_expires_at = utcnow() - timedelta(seconds=1)
    session.commit()

    assert queue.reap_expired_leases(session) == 1
    recovered = queue.claim(session, worker_id="w2", settings=settings)
    assert recovered is not None
    assert recovered.locked_by == "w2"
    assert recovered.attempts == 2, "the crashed attempt still counts toward the cap"


def test_reaper_ignores_live_leases(session, settings):
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    queue.claim(session, worker_id="w1", settings=settings)
    assert queue.reap_expired_leases(session) == 0


def test_queue_stats(session, settings):
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    queue.enqueue(session, tenant_id="t1", kind="prune_trash", settings=settings)
    session.commit()
    queue.claim(session, worker_id="w1", settings=settings)

    stats = queue.queue_stats(session)
    assert stats["queued"] == 1
    assert stats["running"] == 1


# --------------------------------------------------------------------------
# Contention
# --------------------------------------------------------------------------

def test_concurrent_workers_never_double_claim(db, settings):
    """The claim must be atomic: N workers racing over M jobs yields exactly M
    claims with no duplicates. This is the test that would fail under a
    check-then-act (SELECT then UPDATE) implementation."""
    job_count = 25
    worker_count = 6

    with db() as setup:
        for i in range(job_count):
            queue.enqueue(setup, tenant_id="t1", kind="reindex", payload={"i": i}, settings=settings)
        setup.commit()

    claimed_ids: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(worker_count)
    errors: list[Exception] = []

    def worker(worker_id: str):
        try:
            barrier.wait(timeout=10)
            with db() as s:
                while True:
                    job = queue.claim(s, worker_id=worker_id, settings=settings)
                    if job is None:
                        return
                    with lock:
                        claimed_ids.append(job.id)
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"worker errors: {errors}"
    assert len(claimed_ids) == len(set(claimed_ids)), "a job was claimed twice"
    assert len(claimed_ids) == job_count, f"expected {job_count} claims, got {len(claimed_ids)}"

    with db() as verify:
        stats = queue.queue_stats(verify)
        assert stats["running"] == job_count
        assert stats["queued"] == 0


def test_claimed_job_is_invisible_to_other_workers(session, settings):
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()

    first = queue.claim(session, worker_id="w1", settings=settings)
    second = queue.claim(session, worker_id="w2", settings=settings)

    assert first is not None
    assert second is None
