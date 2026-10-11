"""Ingest worker process.

Run one or many; the atomic claim in aka.ingest.queue makes horizontal scaling
safe. Handles SIGTERM by finishing the current job before exiting, so a rolling
restart never abandons in-flight work.

On a laptop the same loop runs as a thread inside the API process
(:class:`EmbeddedWorker`), which keeps a single process owning the embedded
Chroma store.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.db.models import Job, JobStatus
from aka.db.session import init_db, session_scope
from aka.ingest import queue
from aka.llm.faults import build_client
from aka.obs.logging import bind_request_context, clear_request_context, get_logger, setup_logging
from aka.obs.metrics import ingest_jobs
from aka.obs.telemetry import setup_telemetry, stage
from aka.settings import Settings, get_settings

log = get_logger("aka.worker")

MAINTENANCE_INTERVAL_S = 60.0


@dataclass
class JobContext:
    settings: Settings
    client: Any  # model client, for handlers that embed


class Deferred(Exception):
    """Raised by a handler that cannot start yet. The job is re-queued
    without spending an attempt."""

    def __init__(self, reason: str, delay_s: float = 5.0):
        super().__init__(reason)
        self.delay_s = delay_s


JobHandler = Callable[[Session, Job, JobContext], dict]
_HANDLERS: dict[str, JobHandler] = {}


def handler(kind: str) -> Callable[[JobHandler], JobHandler]:
    def register(fn: JobHandler) -> JobHandler:
        _HANDLERS[kind] = fn
        return fn

    return register


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

def _earlier_running_build(session: Session, job: Job) -> str | None:
    """Another reindex for this tenant claimed before this one, if any.

    Two concurrent blue/green builds would race to activate, and the one that
    scanned older state could win. Claim time (``updated_at`` at claim, pinned
    by the lease heartbeat) breaks ties so exactly one of two simultaneous
    claims proceeds.
    """
    others = session.execute(
        select(Job.id, Job.updated_at).where(
            Job.kind == "reindex",
            Job.tenant_id == job.tenant_id,
            Job.status == JobStatus.RUNNING,
            Job.id != job.id,
        )
    ).all()
    mine = (job.updated_at, job.id)
    for other_id, other_claimed in others:
        if (other_claimed, other_id) < mine:
            return other_id
    return None


@handler("reindex")
def handle_reindex(session: Session, job: Job, ctx: JobContext) -> dict:
    """Rebuild the tenant's index incrementally and swap it in."""
    from aka.ingest.pipeline import IncrementalIngestor

    blocker = _earlier_running_build(session, job)
    if blocker is not None:
        raise Deferred(f"reindex {blocker} is already running for this tenant")

    ingestor = IncrementalIngestor(session, ctx.settings, client=ctx.client)
    report = ingestor.run(
        job.tenant_id,
        force_full=bool(job.payload.get("force_full")),
        reap_stale=True,
    )

    if not (report.swapped or report.noop) and report.scanned > 0:
        # The build produced nothing servable; make the job fail so it retries
        # rather than reporting success on a no-op.
        raise RuntimeError(f"index not swapped: {'; '.join(report.errors) or 'unknown'}")
    return report.as_dict()


@handler("prune_trash")
def handle_prune_trash(session: Session, job: Job, ctx: JobContext) -> dict:
    """Delete trashed files whose undo window has closed."""
    from datetime import timedelta
    from pathlib import Path

    from aka.db.models import Document, DocumentStatus, utcnow

    cutoff = utcnow() - timedelta(hours=ctx.settings.UNDO_WINDOW_HOURS)
    purged = 0

    docs = session.scalars(
        select(Document).where(
            Document.tenant_id == job.tenant_id,
            Document.status == DocumentStatus.DELETED,
            Document.deleted_at.is_not(None),
            Document.deleted_at < cutoff,
        )
    )
    for doc in docs:
        if doc.trash_path:
            path = Path(doc.trash_path)
            if path.exists() and path.is_file():
                path.unlink()
        session.delete(doc)
        purged += 1

    session.flush()
    return {"purged": purged}


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------

class Worker:
    def __init__(
        self,
        settings: Settings | None = None,
        worker_id: str | None = None,
        client: Any = None,
    ):
        self.settings = settings or get_settings()
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
        self.ctx = JobContext(settings=self.settings, client=client or build_client(self.settings))
        self._stop = threading.Event()
        self._draining = False
        self._last_maintenance = 0.0

    def request_stop(self, *_args) -> None:
        if not self._draining:
            log.info("worker.draining", worker=self.worker_id)
            self._draining = True
        self._stop.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)

    # --- lease heartbeat ----------------------------------------------------
    def _heartbeat(self, job_id: str, done: threading.Event) -> None:
        lease = self.settings.INGEST_LEASE_SECONDS
        interval = max(1.0, lease / 3)
        while not done.wait(interval):
            try:
                with session_scope(self.settings) as s:
                    if not queue.extend_lease(s, job_id, worker_id=self.worker_id, lease_seconds=lease):
                        log.warning("worker.lease_lost", job_id=job_id)
                        return
            except Exception:
                log.exception("worker.heartbeat_failed", job_id=job_id)

    def run_once(self) -> bool:
        """Claim and execute at most one job. Returns True if work was done."""
        with session_scope(self.settings) as session:
            queue.reap_expired_leases(session)
            job = queue.claim(
                session, worker_id=self.worker_id, kinds=list(_HANDLERS), settings=self.settings
            )
            if job is None:
                return False

            fn = _HANDLERS.get(job.kind)
            bind_request_context(job_id=job.id, job_kind=job.kind, tenant_id=job.tenant_id)
            done = threading.Event()
            beat = threading.Thread(
                target=self._heartbeat, args=(job.id, done), name=f"lease-{job.id[:8]}", daemon=True
            )
            beat.start()
            try:
                if fn is None:
                    # Unknown kind is a permanent error — burn the attempts
                    # immediately instead of retrying a job nobody can run.
                    job.attempts = job.max_attempts
                    queue.fail(session, job, f"no handler registered for kind '{job.kind}'")
                    return True

                started = time.perf_counter()
                with stage(
                    f"job.{job.kind}",
                    **{"aka.job_id": job.id, "aka.tenant_id": job.tenant_id},
                ):
                    result = fn(session, job, self.ctx)
                elapsed = time.perf_counter() - started
                queue.complete(session, job, {**result, "duration_s": round(elapsed, 3)})
            except Deferred as exc:
                session.rollback()
                queue.defer(session, job, delay_s=exc.delay_s, reason=str(exc))
            except Exception as exc:
                session.rollback()
                log.exception("worker.job_failed", job_id=job.id, kind=job.kind)
                queue.fail(session, job, f"{type(exc).__name__}: {exc}", settings=self.settings)
            finally:
                done.set()
                beat.join(timeout=5)
                clear_request_context()
                queue.queue_stats(session)
            return True

    def maintenance(self) -> None:
        """Housekeeping that needs no job: expire stale approvals."""
        from aka.hitl.approvals import expire_stale

        with session_scope(self.settings) as session:
            expire_stale(session)

    def run_forever(self) -> None:
        log.info("worker.started", worker=self.worker_id, handlers=sorted(_HANDLERS))
        idle_polls = 0
        while not self._stop.is_set():
            try:
                if time.monotonic() - self._last_maintenance > MAINTENANCE_INTERVAL_S:
                    self._last_maintenance = time.monotonic()
                    self.maintenance()
                did_work = self.run_once()
            except Exception:
                # A failure in the loop itself (DB down) must not kill the
                # worker; back off and keep trying.
                log.exception("worker.loop_error", worker=self.worker_id)
                did_work = False

            if did_work:
                idle_polls = 0
            else:
                idle_polls += 1
                # Gentle backoff while idle so an empty queue doesn't spin the
                # DB, capped so latency on a new job stays bounded.
                delay = min(self.settings.INGEST_POLL_INTERVAL_S * (1 + idle_polls / 10), 5.0)
                self._stop.wait(delay)

        log.info("worker.stopped", worker=self.worker_id)


class EmbeddedWorker:
    """The worker loop on a background thread of the API process."""

    def __init__(self, settings: Settings, client: Any = None):
        self.worker = Worker(settings, client=client)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.worker.run_forever, name="aka-embedded-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the loop to stop after the current job and wait for it."""
        self.worker.request_stop()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None


def main() -> None:  # pragma: no cover - entrypoint
    settings = get_settings()
    setup_logging(settings)
    setup_telemetry(settings)
    init_db(settings)

    worker = Worker(settings)
    worker.install_signal_handlers()
    ingest_jobs.labels(status="worker_start").inc()
    worker.run_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
