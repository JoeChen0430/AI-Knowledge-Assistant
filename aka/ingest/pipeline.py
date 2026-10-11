"""Incremental ingestion.

The old pipeline deleted the whole collection and re-embedded every document on
any change, which cost O(corpus) per edit and left the knowledge base empty for
the duration. This one:

1. Scans DATA_DIR and diffs file digests against the ``documents`` table.
2. Parses only files whose digest changed.
3. Builds a new IndexVersion, copying embeddings for unchanged documents
   straight from the active collection and embedding only what is new.
4. Smoke-checks the new collection and swaps it in atomically.

Failure at any point leaves the previous version ACTIVE and serving.

Transactions are kept short on purpose. Embedding is the slow part (minutes
on a large corpus) and holding a write transaction across it would lock
SQLite for the whole build, so every chat turn that writes a message during a
rebuild would fail with "database is locked". A run is therefore split:

    plan      read-only   scan, diff digests, parse changed files
    begin     short txn   record the BUILDING version
    populate  no writes   copy reused vectors, embed the rest into the new
                          collection, collect chunk rows in memory
    finalize  short txn   smoke-check, write documents + chunks, flip ACTIVE

Document hashes only change in the final transaction, so a build that dies
part-way leaves every row as it was and the next run redoes the work.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from aka.db.models import Chunk, Document, DocumentStatus, IndexVersion, new_id, utcnow
from aka.ingest.chunker import ParsedChunk, parse_document
from aka.ingest.hashing import chunk_id as make_chunk_id
from aka.ingest.hashing import file_digest
from aka.ingest.vectorstore import IndexManager, VectorStore
from aka.llm.embedding import ResilientEmbedder
from aka.llm.faults import build_client
from aka.obs.logging import get_logger
from aka.obs.metrics import ingest_chunks, ingest_duration
from aka.obs.telemetry import stage
from aka.security.paths import is_within, relative_label
from aka.settings import Settings, get_settings

log = get_logger("aka.ingest")

EMBED_BATCH = 16
# Keeps IN (...) lists under SQLite's bound-parameter limit.
_DELETE_BATCH = 500
# Vectors moved per Chroma get/upsert when reusing an unchanged corpus.
_COPY_BATCH = 2000


@dataclass
class IngestReport:
    tenant_id: str
    scanned: int = 0
    unchanged: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    failed: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    index_version: str | None = None
    swapped: bool = False
    # Nothing changed and the active index is intact: no new version built.
    noop: bool = False
    duration_s: float = 0.0
    # Where the time went: plan (scan + parse), embed, copy, finalize.
    timings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    errors: list[str] = field(default_factory=list)

    @property
    def reuse_ratio(self) -> float:
        total = self.chunks_embedded + self.chunks_reused
        return self.chunks_reused / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "scanned": self.scanned,
            "unchanged": self.unchanged,
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "failed": self.failed,
            "chunks_embedded": self.chunks_embedded,
            "chunks_reused": self.chunks_reused,
            "reuse_ratio": round(self.reuse_ratio, 4),
            "index_version": self.index_version,
            "swapped": self.swapped,
            "noop": self.noop,
            "duration_s": round(self.duration_s, 3),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "errors": self.errors[:20],
        }


@dataclass
class _Plan:
    """What a run intends to do, computed without writing anything."""

    live: list[Document] = field(default_factory=list)       # docs in the new version
    parsed: dict[str, list[ParsedChunk]] = field(default_factory=dict)  # doc id -> chunks
    changes: dict[str, tuple[str, int]] = field(default_factory=dict)   # doc id -> (digest, size)
    new_docs: list[Document] = field(default_factory=list)   # transient until finalize
    revived: list[Document] = field(default_factory=list)    # soft-deleted, now back
    removed: list[Document] = field(default_factory=list)
    failed: list[Document] = field(default_factory=list)     # current file unparseable
    dropped: list[Document] = field(default_factory=list)    # failed and nothing reusable


def scan_data_dir(settings: Settings) -> dict[str, Path]:
    """Map rel_path -> absolute path for every ingestable file."""
    root = Path(settings.DATA_DIR)
    if not root.exists():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in settings.ALLOWED_EXTENSIONS:
            continue
        # rglob can surface symlinked files from outside the root.
        if not is_within(root, path):
            log.warning("ingest.skipped_outside_root", path=str(path))
            continue
        found[relative_label(path, root)] = path
    return found


class IncrementalIngestor:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        client=None,
    ):
        self.session = session
        self.settings = settings or get_settings()
        self.client = client or build_client(self.settings)
        self.embedder = ResilientEmbedder(self.client, self.settings)
        self.index = IndexManager(session, self.settings)

    def run(
        self, tenant_id: str, *, force_full: bool = False, reap_stale: bool = False
    ) -> IngestReport:
        """Bring the tenant's index in line with DATA_DIR.

        ``reap_stale`` fails BUILDING versions left by a crashed build; only the
        worker passes it, because only the worker knows no other build for this
        tenant is in flight.
        """
        started = time.perf_counter()
        report = IngestReport(tenant_id=tenant_id)

        with stage("ingest.run", **{"aka.tenant_id": tenant_id}):
            if reap_stale:
                self.index.reap_stale_builds(tenant_id)
                self.session.commit()

            previous = self.index.active_version(tenant_id)
            if previous is not None and previous.embed_model not in ("", self.settings.EMBED_MODEL):
                # Vectors from another embedding model live in a different
                # space; copying them across would silently wreck retrieval.
                log.warning(
                    "ingest.embed_model_changed",
                    previous=previous.embed_model,
                    current=self.settings.EMBED_MODEL,
                )
                force_full = True

            t0 = time.perf_counter()
            plan = self._plan(tenant_id, report, force_full=force_full)
            report.timings["plan_s"] += time.perf_counter() - t0

            if not plan.live:
                self._finalize_empty(tenant_id, plan, report)
                return self._finish(report, started)

            if not force_full and self._is_noop(tenant_id, plan, previous):
                report.noop = True
                report.index_version = previous.id
                report.chunks_reused = previous.chunk_count
                self.session.rollback()  # end the read transaction
                log.info("ingest.noop", tenant=tenant_id, version=previous.id)
                return self._finish(report, started)

            previous_store = None if force_full else self.index.active_store(tenant_id)
            version = self.index.begin_build(
                tenant_id, notes=f"added={report.added} updated={report.updated} removed={report.removed}"
            )
            report.index_version = version.id
            self.session.commit()  # BUILDING row is visible; no write lock from here on

            new_store = self.index.store_for(version)
            try:
                total, pending, replace = self._populate(
                    tenant_id, plan, previous_store, new_store, report
                )
            except Exception as exc:
                self.session.rollback()
                self.index.abandon(version, f"build failed: {exc}")
                self.session.commit()
                report.errors.append(f"build failed: {exc}")
                log.exception("ingest.build_failed", tenant=tenant_id)
                self._finish(report, started)
                raise

            t0 = time.perf_counter()
            self._finalize(tenant_id, version, plan, total, pending, replace, report)
            report.timings["finalize_s"] += time.perf_counter() - t0

        return self._finish(report, started)

    # ----------------------------------------------------------------------
    def _finish(self, report: IngestReport, started: float) -> IngestReport:
        report.duration_s = time.perf_counter() - started
        ingest_duration.observe(report.duration_s)
        log.info("ingest.completed", **report.as_dict())
        return report

    def _is_noop(self, tenant_id: str, plan: _Plan, previous: IndexVersion | None) -> bool:
        """Nothing changed on disk and the active collection is intact, so a
        rebuild would only copy every vector into an identical collection."""
        if previous is None or previous.chunk_count <= 0:
            return False
        if plan.changes or plan.removed or plan.failed:
            return False
        store = self.index.active_store(tenant_id)
        return store is not None and store.count() == previous.chunk_count

    # --- plan: read-only ----------------------------------------------------
    def _plan(self, tenant_id: str, report: IngestReport, *, force_full: bool) -> _Plan:
        plan = _Plan()
        on_disk = scan_data_dir(self.settings)
        report.scanned = len(on_disk)

        rows = list(self.session.scalars(select(Document).where(Document.tenant_id == tenant_id)))
        known = {d.rel_path: d for d in rows if d.status is not DocumentStatus.DELETED}
        deleted = {d.rel_path: d for d in rows if d.status is DocumentStatus.DELETED}

        for rel_path, path in on_disk.items():
            doc = known.get(rel_path)
            if doc is not None and doc.status is DocumentStatus.QUARANTINED:
                # Released only through the approval queue, never by editing
                # the file until its hash changes.
                continue

            try:
                digest = file_digest(path)
                size = path.stat().st_size
            except OSError as exc:
                report.failed += 1
                report.errors.append(f"{rel_path}: unreadable ({exc})")
                if doc is not None:
                    plan.live.append(doc)  # keep serving what we had
                continue

            if doc is not None and doc.content_hash == digest and not force_full:
                report.unchanged += 1
                plan.live.append(doc)
                continue

            try:
                chunks = parse_document(path, self.settings)
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{rel_path}: parse failed ({exc})")
                log.warning("ingest.parse_failed", path=rel_path, error=str(exc))
                if doc is not None:
                    # The last good version stays searchable; the hash is not
                    # updated, so the next run retries the parse.
                    plan.failed.append(doc)
                    plan.live.append(doc)
                continue

            if doc is None:
                doc = deleted.get(rel_path)
                if doc is not None:
                    # Same path as a soft-deleted row: revive it rather than
                    # violate the (tenant, rel_path) unique constraint.
                    plan.revived.append(doc)
                else:
                    doc = Document(
                        id=new_id(),
                        tenant_id=tenant_id,
                        rel_path=rel_path,
                        content_hash=digest,
                        size_bytes=size,
                        mime_type=path.suffix.lower().lstrip("."),
                        status=DocumentStatus.PENDING,
                    )
                    plan.new_docs.append(doc)
                report.added += 1
            else:
                report.updated += 1

            plan.changes[doc.id] = (digest, size)
            plan.parsed[doc.id] = chunks
            plan.live.append(doc)

        # Documents whose file disappeared are soft-deleted, not dropped,
        # so an accidental removal is still recoverable.
        for rel_path, doc in known.items():
            if rel_path not in on_disk and doc.status is not DocumentStatus.QUARANTINED:
                plan.removed.append(doc)
                report.removed += 1

        return plan

    # --- populate: no DB writes --------------------------------------------
    def _populate(
        self,
        tenant_id: str,
        plan: _Plan,
        previous_store: VectorStore | None,
        new_store: VectorStore,
        report: IngestReport,
    ) -> tuple[int, list[Chunk], set[str]]:
        """Fill the new collection, reusing vectors wherever the document is
        unchanged. Returns (chunk total, chunk rows to write, doc ids whose
        chunk rows are replaced)."""
        total = 0
        pending: list[Chunk] = []
        replace: set[str] = set(plan.changes)

        # Unchanged documents first: their chunk rows are still valid, so the
        # vectors are copied across without touching the embedding model, in
        # large batches spanning many documents (per-document round trips
        # dominated update time on a big corpus).
        unchanged = [d for d in plan.live if d.id not in plan.parsed]
        copied_ok: set[str] = set()
        if previous_store is not None and unchanged:
            t0 = time.perf_counter()
            copied_ok, reused = self._copy_unchanged(unchanged, previous_store, new_store)
            report.timings["copy_s"] += time.perf_counter() - t0
            report.chunks_reused += reused
            ingest_chunks.labels(action="reused").inc(reused)
            total += reused

        for doc in plan.live:
            if doc.id in copied_ok:
                continue
            chunks = plan.parsed.get(doc.id)

            if chunks is None:
                # Unchanged, but its vectors could not be reused (no previous
                # index, or ids missing from it): rebuild rather than drop.
                try:
                    chunks = parse_document(Path(self.settings.DATA_DIR) / doc.rel_path, self.settings)
                except Exception as exc:
                    # Nothing reusable and the file no longer parses: leave
                    # it out of this version rather than fail the whole build.
                    report.errors.append(f"{doc.rel_path}: unrecoverable ({exc})")
                    if doc not in plan.failed:
                        plan.failed.append(doc)
                    plan.dropped.append(doc)
                    continue
                replace.add(doc.id)

            written, rows = self._embed_document(tenant_id, doc, chunks, new_store, report)
            pending.extend(rows)
            total += written

        return total, pending, replace

    def _copy_unchanged(
        self, docs: list[Document], source: VectorStore, target: VectorStore
    ) -> tuple[set[str], int]:
        """Copy every unchanged document's vectors. Returns the ids of
        documents copied completely, and the number of chunks copied for them.

        A document is reused only if *all* of its chunk ids came across; a
        partial copy is deleted from the target so the caller re-embeds it.
        """
        doc_ids = [d.id for d in docs]
        by_doc: dict[str, list[str]] = defaultdict(list)
        # Column query, not ORM rows: nothing enters the identity map.
        for start in range(0, len(doc_ids), _DELETE_BATCH):
            window = doc_ids[start : start + _DELETE_BATCH]
            for cid, did in self.session.execute(
                select(Chunk.id, Chunk.document_id).where(Chunk.document_id.in_(window))
            ):
                by_doc[did].append(cid)

        wanted = [cid for did in doc_ids for cid in by_doc.get(did, [])]
        present = target.copy_vectors_from(source, wanted, batch=_COPY_BATCH) if wanted else set()

        complete: set[str] = set()
        reused = 0
        for doc in docs:
            ids = by_doc.get(doc.id, [])
            if ids and all(cid in present for cid in ids):
                complete.add(doc.id)
                reused += len(ids)
                continue
            if ids:
                # Vectors missing from the old collection must be rebuilt
                # rather than silently dropped.
                log.warning(
                    "ingest.reuse_incomplete",
                    document=doc.rel_path,
                    expected=len(ids),
                    copied=sum(1 for cid in ids if cid in present),
                )
                target.delete([cid for cid in ids if cid in present])
        return complete, reused

    def _embed_document(
        self,
        tenant_id: str,
        doc: Document,
        chunks: list[ParsedChunk],
        store: VectorStore,
        report: IngestReport,
    ) -> tuple[int, list[Chunk]]:
        rows: list[Chunk] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            texts = [c.text for c in batch]

            t0 = time.perf_counter()
            with stage("ingest.embed", **{"aka.batch_size": len(batch)}):
                vectors = self.embedder.embed(texts)
            report.timings["embed_s"] += time.perf_counter() - t0

            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding count mismatch for {doc.rel_path}: "
                    f"{len(vectors)} vectors for {len(batch)} chunks"
                )

            ids, metadatas = [], []
            for parsed_chunk in batch:
                cid = make_chunk_id(tenant_id, doc.id, parsed_chunk.ordinal, parsed_chunk.text)
                ids.append(cid)
                metadatas.append(
                    {
                        "document_id": doc.id,
                        "tenant_id": tenant_id,
                        "rel_path": doc.rel_path,
                        "ordinal": parsed_chunk.ordinal,
                        "chunk_hash": parsed_chunk.hash,
                    }
                )
                rows.append(
                    Chunk(
                        id=cid,
                        document_id=doc.id,
                        tenant_id=tenant_id,
                        ordinal=parsed_chunk.ordinal,
                        chunk_hash=parsed_chunk.hash,
                        token_count=parsed_chunk.token_count,
                        text_preview=parsed_chunk.text[:300],
                    )
                )

            store.upsert(ids, vectors, texts, metadatas)
            report.chunks_embedded += len(batch)
            ingest_chunks.labels(action="embedded").inc(len(batch))

        return len(rows), rows

    # --- finalize: one short write transaction -------------------------------
    def _finalize(
        self,
        tenant_id: str,
        version: IndexVersion,
        plan: _Plan,
        total: int,
        pending: list[Chunk],
        replace: set[str],
        report: IngestReport,
    ) -> None:
        ok, detail = self.index.smoke_check(version, expected_chunks=total)
        if not ok:
            # Nothing about the documents is written: the old index keeps
            # serving and the next run tries the same changes again.
            self.index.reject(version, detail)
            self.session.commit()
            report.errors.append(f"index activation rejected by smoke check: {detail}")
            return

        self._write_documents(plan, pending, replace)
        version.doc_count = len(plan.live) - len(plan.dropped)

        if not self.index.activate(version, expected_chunks=total):
            # Smoke check passed a moment ago; refuse to half-apply anyway.
            self.session.rollback()
            self.index.reject(version, "smoke check failed at activation")
            self.session.commit()
            report.errors.append("index activation rejected at swap time")
            return

        failed_ids = {d.id for d in plan.failed}
        for doc in plan.live:
            if doc.id not in failed_ids:
                doc.status = DocumentStatus.INDEXED
        self.index.prune_retired(tenant_id, keep=1)
        self.session.commit()
        report.swapped = True

    def _write_documents(self, plan: _Plan, pending: list[Chunk], replace: set[str]) -> None:
        session = self.session
        by_id = {d.id: d for d in plan.live}

        for doc in plan.new_docs:
            session.add(doc)
        for doc in plan.revived:
            doc.deleted_at = None
            doc.trash_path = None
        for doc_id, (digest, size) in plan.changes.items():
            doc = by_id[doc_id]
            doc.content_hash = digest
            doc.size_bytes = size
            doc.status = DocumentStatus.PENDING
            doc.updated_at = utcnow()
        session.flush()  # documents before the chunks that reference them

        replace_ids = list(replace)
        for start in range(0, len(replace_ids), _DELETE_BATCH):
            window = replace_ids[start : start + _DELETE_BATCH]
            session.execute(
                delete(Chunk)
                .where(Chunk.document_id.in_(window))
                .execution_options(synchronize_session=False)
            )
        session.add_all(pending)

        now = utcnow()
        for doc in plan.removed:
            doc.status = DocumentStatus.DELETED
            doc.deleted_at = now
        for doc in plan.failed:
            doc.status = DocumentStatus.FAILED
        session.flush()

    def _finalize_empty(self, tenant_id: str, plan: _Plan, report: IngestReport) -> None:
        """No live documents: record removals and take the index out of
        service if the corpus was emptied on purpose."""
        now = utcnow()
        for doc in plan.removed:
            doc.status = DocumentStatus.DELETED
            doc.deleted_at = now
        if plan.removed:
            self.index.retire_active(tenant_id, "every document was removed")
        else:
            log.warning("ingest.no_documents", tenant=tenant_id)
        self.session.commit()
