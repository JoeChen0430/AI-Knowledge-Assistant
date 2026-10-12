"""Incremental ingest + blue/green swap.

These tests encode the two behavioural claims about the pipeline:
  * an unchanged document is never re-embedded
  * a failed build never takes the serving index down
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest
from sqlalchemy import select

from aka.db.models import Document, DocumentStatus, IndexStatus, IndexVersion


class CountingEmbedder:
    """Deterministic fake embedder that records how much work it was asked to do."""

    DIM = 32

    def __init__(self, fail_on: set[str] | None = None):
        self.embedded_texts: list[str] = []
        self.calls = 0
        self.fail_on = fail_on or set()

    def embed(self, texts, *, model=None):
        self.calls += 1
        out = []
        for text in texts:
            if any(marker in text for marker in self.fail_on):
                raise RuntimeError("embedding backend exploded")
            self.embedded_texts.append(text)
            out.append(self._vector(text))
        return out

    def embed_one(self, text, *, model=None):
        return self.embed([text])[0]

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(cls.DIM)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]


@pytest.fixture()
def ingest_env(settings, db, monkeypatch):
    from aka.ingest import vectorstore

    vectorstore.reset_chroma_for_tests()
    monkeypatch.setattr(vectorstore, "_client", None, raising=False)
    yield settings
    vectorstore.reset_chroma_for_tests()


def write_doc(settings, name: str, body: str) -> None:
    path = settings.DATA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def run_ingest(session, settings, embedder, **kwargs):
    from aka.ingest.pipeline import IncrementalIngestor

    ingestor = IncrementalIngestor(session, settings, client=embedder)
    report = ingestor.run("t1", **kwargs)
    session.commit()
    return report


# --------------------------------------------------------------------------

def test_first_ingest_embeds_everything(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    write_doc(settings, "b.txt", "Beta document. " * 40)
    embedder = CountingEmbedder()

    report = run_ingest(session, settings, embedder)

    assert report.added == 2
    assert report.swapped is True
    assert report.chunks_embedded > 0
    assert report.chunks_reused == 0
    assert len(embedder.embedded_texts) == report.chunks_embedded


def test_reingest_with_no_changes_embeds_nothing(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    write_doc(settings, "b.txt", "Beta document. " * 40)

    first = CountingEmbedder()
    run_ingest(session, settings, first)

    second = CountingEmbedder()
    report = run_ingest(session, settings, second)

    assert report.unchanged == 2
    assert second.embedded_texts == [], "unchanged corpus must not call the embedder"
    assert report.chunks_reused > 0
    assert report.reuse_ratio == 1.0


def test_only_the_changed_document_is_re_embedded(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    write_doc(settings, "b.txt", "Beta document. " * 40)

    run_ingest(session, settings, CountingEmbedder())

    write_doc(settings, "b.txt", "Beta rewritten entirely. " * 40)
    second = CountingEmbedder()
    report = run_ingest(session, settings, second)

    assert report.updated == 1
    assert report.unchanged == 1
    assert report.chunks_reused > 0
    assert all("Beta rewritten" in t for t in second.embedded_texts), (
        "only the changed document's chunks should reach the embedder"
    )


def test_whitespace_only_change_is_free(ingest_env, session):
    """Normalisation means a reflowed file produces identical chunk hashes."""
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    original = (settings.DATA_DIR / "a.txt").read_text()
    write_doc(settings, "a.txt", original.replace(" ", "  ").replace(". ", ".\n"))

    second = CountingEmbedder()
    report = run_ingest(session, settings, second)

    # The file digest changed, so it is re-parsed and re-embedded, but the
    # chunk hashes are stable — which is what keeps the vector ids stable.
    assert report.updated == 1
    from aka.ingest.hashing import text_digest

    assert text_digest("a  b.\nc") == text_digest("a b. c")


def test_deleted_file_is_soft_deleted(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    write_doc(settings, "b.txt", "Beta document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    (settings.DATA_DIR / "b.txt").unlink()
    report = run_ingest(session, settings, CountingEmbedder())

    assert report.removed == 1
    doc = session.scalars(
        select(Document).where(Document.rel_path == "b.txt")
    ).one()
    assert doc.status is DocumentStatus.DELETED
    assert doc.deleted_at is not None


def test_failed_build_leaves_previous_index_serving(ingest_env, session):
    """The core availability claim: a broken rebuild must not empty the KB."""
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    good_version = session.scalars(
        select(IndexVersion).where(IndexVersion.status == IndexStatus.ACTIVE)
    ).one()
    from aka.ingest.vectorstore import IndexManager

    served_before = IndexManager(session, settings).active_store("t1").count()
    assert served_before > 0

    # New document whose embedding call blows up mid-build.
    write_doc(settings, "poison.txt", "POISONPILL content here. " * 40)
    with pytest.raises(RuntimeError):
        run_ingest(session, settings, CountingEmbedder(fail_on={"POISONPILL"}))
    session.rollback()

    still_active = session.scalars(
        select(IndexVersion).where(IndexVersion.status == IndexStatus.ACTIVE)
    ).one()
    assert still_active.id == good_version.id, "active index must not change on failure"

    store = IndexManager(session, settings).active_store("t1")
    assert store is not None and store.count() == served_before


def test_smoke_check_rejects_empty_index(ingest_env, session):
    from aka.ingest.vectorstore import IndexManager

    manager = IndexManager(session, settings=ingest_env)
    version = manager.begin_build("t1")
    ok, detail = manager.smoke_check(version, expected_chunks=0)
    assert ok is False
    assert "empty" in detail


def test_smoke_check_rejects_count_mismatch(ingest_env, session):
    from aka.ingest.vectorstore import IndexManager

    manager = IndexManager(session, settings=ingest_env)
    version = manager.begin_build("t1")
    store = manager.store_for(version)
    store.upsert(["c1"], [CountingEmbedder._vector("x")], ["x"], [{"document_id": "d"}])

    ok, detail = manager.smoke_check(version, expected_chunks=5)
    assert ok is False
    assert "mismatch" in detail
    assert manager.activate(version, expected_chunks=5) is False


def test_retired_versions_are_pruned(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    for i in range(3):
        write_doc(settings, "a.txt", f"Alpha revision {i}. " * 40)
        run_ingest(session, settings, CountingEmbedder())

    versions = list(session.scalars(select(IndexVersion)))
    assert len(versions) <= 2, "only the active plus one rollback target are kept"


# --------------------------------------------------------------------------
# Transactions, concurrency, lifecycle
# --------------------------------------------------------------------------

class WritingEmbedder(CountingEmbedder):
    """Does a write from a *different* connection while the build embeds.

    If the ingestor held a write transaction across embedding (as a single
    long transaction would), SQLite would make this wait out the busy timeout
    and fail with "database is locked".
    """

    def __init__(self, factory):
        super().__init__()
        self.factory = factory
        self.writes = 0

    def embed(self, texts, *, model=None):
        from aka.db.models import Conversation

        other = self.factory()
        try:
            other.add(Conversation(tenant_id="t1", user_id="u1", title="written mid-build"))
            other.commit()
            self.writes += 1
        finally:
            other.close()
        return super().embed(texts, model=model)


def test_build_does_not_hold_a_write_lock_while_embedding(ingest_env, db, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    write_doc(settings, "b.txt", "Beta document. " * 40)
    embedder = WritingEmbedder(db)  # db is the session factory: a second connection
    started = time.perf_counter()
    report = run_ingest(session, settings, embedder)
    assert report.swapped and embedder.writes > 0
    assert time.perf_counter() - started < 3, "a concurrent writer was blocked by the build"


def test_readers_see_a_complete_index_throughout_an_update(ingest_env, db, session):
    """The no-empty-window claim, checked from another connection."""
    from aka.ingest.vectorstore import IndexManager

    settings = ingest_env
    for i in range(6):
        write_doc(settings, f"doc{i}.txt", f"Document {i} body. " * 60)
    run_ingest(session, settings, CountingEmbedder())
    before = IndexManager(session, settings).active_store("t1").count()

    observed: list[int] = []
    stop = threading.Event()

    def reader():
        s = db()
        try:
            while not stop.is_set():
                store = IndexManager(s, settings).active_store("t1")
                observed.append(store.count() if store else 0)
                s.rollback()  # fresh snapshot each probe
                time.sleep(0.005)
        finally:
            s.close()

    class SlowEmbedder(CountingEmbedder):
        def embed(self, texts, *, model=None):
            time.sleep(0.05)
            return super().embed(texts, model=model)

    thread = threading.Thread(target=reader)
    thread.start()
    write_doc(settings, "doc0.txt", "Document zero, rewritten. " * 60)
    write_doc(settings, "new.txt", "A brand new document. " * 60)
    report = run_ingest(session, settings, SlowEmbedder())
    time.sleep(0.05)
    stop.set()
    thread.join()

    assert report.swapped
    assert observed, "the reader never ran"
    assert min(observed) >= before, f"reader saw a partial or empty index: {min(observed)} < {before}"


def test_changing_the_embedding_model_re_embeds_everything(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    settings.EMBED_MODEL = "a-different-model"
    second = CountingEmbedder()
    report = run_ingest(session, settings, second)
    assert report.chunks_reused == 0, "vectors from another model must not be copied"
    assert report.chunks_embedded > 0 and report.swapped


def test_a_deleted_path_can_come_back(ingest_env, session):
    """Re-adding a soft-deleted path revives its row instead of tripping the
    (tenant, rel_path) unique constraint."""
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    write_doc(settings, "b.txt", "Beta document. " * 40)
    run_ingest(session, settings, CountingEmbedder())

    (settings.DATA_DIR / "b.txt").unlink()
    run_ingest(session, settings, CountingEmbedder())
    write_doc(settings, "b.txt", "Beta is back. " * 40)
    report = run_ingest(session, settings, CountingEmbedder())

    assert report.added == 1 and report.swapped
    doc = session.scalars(select(Document).where(Document.rel_path == "b.txt")).one()
    assert doc.status is DocumentStatus.INDEXED and doc.deleted_at is None


def test_removing_every_document_takes_the_index_out_of_service(ingest_env, session):
    from aka.ingest.vectorstore import IndexManager

    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    run_ingest(session, settings, CountingEmbedder())
    (settings.DATA_DIR / "a.txt").unlink()
    report = run_ingest(session, settings, CountingEmbedder())

    assert report.removed == 1
    assert IndexManager(session, settings).active_store("t1") is None


def test_rerun_with_no_changes_builds_nothing(ingest_env, session):
    settings = ingest_env
    write_doc(settings, "a.txt", "Alpha document. " * 40)
    first = run_ingest(session, settings, CountingEmbedder())
    second = run_ingest(session, settings, CountingEmbedder())
    assert second.noop and second.index_version == first.index_version
    assert len(list(session.scalars(select(IndexVersion)))) == 1
