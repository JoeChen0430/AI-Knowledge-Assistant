"""Worker: one build per tenant at a time, lease heartbeat, end-to-end reindex."""

from __future__ import annotations

import time

import pytest
from sqlalchemy import select

from aka.db.models import Job, JobStatus
from aka.ingest import queue
from aka.llm.fake import FakeOllama


@pytest.fixture()
def fresh_chroma():
    from aka.ingest import vectorstore

    vectorstore.reset_chroma_for_tests()
    yield
    vectorstore.reset_chroma_for_tests()


def _reload(session, job_id):
    return session.scalars(
        select(Job).where(Job.id == job_id).execution_options(populate_existing=True)
    ).one()


def test_second_build_for_the_same_tenant_is_deferred(session, settings, db):
    from aka.ingest.worker import Deferred, JobContext, _earlier_running_build, handle_reindex

    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    first = queue.claim(session, worker_id="w1", settings=settings)
    second = queue.claim(session, worker_id="w2", settings=settings)

    assert _earlier_running_build(session, first) is None
    assert _earlier_running_build(session, second) == first.id
    with pytest.raises(Deferred):
        handle_reindex(session, second, JobContext(settings, FakeOllama()))

    queue.defer(session, second, delay_s=0, reason="busy")
    again = _reload(session, second.id)
    assert again.status is JobStatus.QUEUED
    assert again.attempts == 0, "a deferral must not spend an attempt"


def test_other_tenants_are_not_blocked(session, settings, db):
    from aka.ingest.worker import _earlier_running_build

    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    queue.enqueue(session, tenant_id="t2", kind="reindex", settings=settings)
    session.commit()
    queue.claim(session, worker_id="w1", settings=settings)
    other = queue.claim(session, worker_id="w2", settings=settings)
    assert _earlier_running_build(session, other) is None


def test_heartbeat_extends_only_its_own_lease(session, settings, db):
    queue.enqueue(session, tenant_id="t1", kind="reindex", settings=settings)
    session.commit()
    job = queue.claim(session, worker_id="w1", lease_seconds=5, settings=settings)
    lease_before, claimed_at = job.lease_expires_at, job.updated_at

    time.sleep(0.01)
    assert queue.extend_lease(session, job.id, worker_id="w1", lease_seconds=600)
    after = _reload(session, job.id)
    assert after.lease_expires_at > lease_before
    assert after.updated_at == claimed_at, "the claim time stays readable"
    assert not queue.extend_lease(session, job.id, worker_id="intruder", lease_seconds=600)


def test_worker_runs_a_reindex_end_to_end(settings, db, session, fresh_chroma):
    from aka.ingest.vectorstore import IndexManager
    from aka.ingest.worker import Worker

    (settings.DATA_DIR / "a.txt").write_text("Alpha document body. " * 30, encoding="utf-8")
    job = queue.ensure_reindex(session, tenant_id="t1", reason="test", settings=settings)
    session.commit()

    assert Worker(settings, client=FakeOllama()).run_once() is True
    done = _reload(session, job.id)
    assert done.status is JobStatus.SUCCEEDED, done.last_error
    assert done.result["swapped"] is True
    assert IndexManager(session, settings).active_store("t1").count() > 0

    # Nothing changed: the next reindex is a no-op, not a failure.
    queue.ensure_reindex(session, tenant_id="t1", reason="again", settings=settings)
    session.commit()
    Worker(settings, client=FakeOllama()).run_once()
    latest = session.scalars(select(Job).order_by(Job.created_at.desc())).first()
    assert latest.status is JobStatus.SUCCEEDED and latest.result["noop"] is True


def test_ensure_reindex_reuses_a_waiting_job(session, settings, db):
    a = queue.ensure_reindex(session, tenant_id="t1", reason="upload", settings=settings)
    b = queue.ensure_reindex(session, tenant_id="t1", reason="upload", settings=settings)
    assert a.id == b.id
