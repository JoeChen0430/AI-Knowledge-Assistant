"""Chroma access plus blue/green index versioning.

Readers never name a collection directly. They ask for the tenant's ACTIVE
IndexVersion, so a rebuild can populate a fresh collection at full speed while
queries keep hitting the old one, and the swap is a single row update.

The other half of the trick lives in :meth:`VectorStore.copy_vectors_from`:
when a new version is built, chunks whose hash is unchanged have their existing
embeddings copied across instead of re-embedded. Embedding is ~99% of ingest
cost, so an unchanged corpus rebuilds almost for free.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.db.models import IndexStatus, IndexVersion, new_id, utcnow
from aka.obs.logging import get_logger
from aka.obs.metrics import index_swaps
from aka.obs.telemetry import stage
from aka.settings import Settings, get_settings

log = get_logger("aka.vectorstore")

# Cosine, not the Chroma default of L2 — embeddings are direction-normalised,
# and cosine gives a bounded score we can threshold on consistently.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}

_client: chromadb.ClientAPI | None = None
_client_lock = threading.Lock()


def get_chroma(settings: Settings | None = None) -> chromadb.ClientAPI:
    global _client
    settings = settings or get_settings()
    with _client_lock:
        if _client is None:
            if settings.CHROMA_HOST:
                _client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
                )
            else:
                _client = chromadb.PersistentClient(path=str(settings.DB_DIR))
        return _client


def reset_chroma_for_tests() -> None:
    global _client
    with _client_lock:
        _client = None


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    score: float           # cosine similarity in [0, 1]; higher is better
    metadata: dict[str, Any]

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))

    @property
    def source(self) -> str:
        return str(self.metadata.get("rel_path", "unknown"))


class VectorStore:
    """Thin wrapper over one Chroma collection."""

    def __init__(self, collection: Collection):
        self.collection = collection

    @property
    def name(self) -> str:
        return self.collection.name

    def count(self) -> int:
        return self.collection.count()

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[_clean_metadata(m) for m in metadatas],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self.collection.delete(ids=list(ids))

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        found = self.collection.get(ids=list(ids), include=[])
        return set(found.get("ids") or [])

    def copy_vectors_from(
        self, source: "VectorStore", ids: Sequence[str], *, batch: int = 2000
    ) -> set[str]:
        """Move already-computed embeddings into this collection. Returns the
        ids actually found in ``source`` and copied.

        Batched because Chroma materialises the whole result set in memory;
        large batches because each get/upsert round trip has a fixed cost.
        """
        ids = list(ids)
        copied: set[str] = set()
        for start in range(0, len(ids), batch):
            window = ids[start : start + batch]
            got = source.collection.get(
                ids=window, include=["embeddings", "documents", "metadatas"]
            )
            got_ids = got.get("ids") or []
            if not got_ids:
                continue
            self.collection.upsert(
                ids=got_ids,
                embeddings=got["embeddings"],
                documents=got["documents"],
                metadatas=got["metadatas"],
            )
            copied.update(got_ids)
        return copied

    def query(
        self,
        embedding: Sequence[float],
        *,
        k: int = 8,
        where: dict | None = None,
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(k, max(1, self.count())),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = _rows(result, "ids")
        docs = _rows(result, "documents")
        metas = _rows(result, "metadatas")
        dists = _rows(result, "distances")
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    text=doc or "",
                    # Chroma returns cosine *distance*; clamp because floating
                    # point can put it a hair outside [0, 2].
                    score=max(0.0, min(1.0, 1.0 - float(dist) )),
                    metadata=dict(meta or {}),
                )
            )
        return hits

    def keyword_search(
        self,
        terms: Sequence[str],
        *,
        k: int = 8,
        where: dict | None = None,
        scan_limit: int = 200,
    ) -> list[SearchHit]:
        """Substring retrieval for when no query vector is available.

        The degraded path for an embedding outage: worse ranking than vector
        search, but it keeps the answer grounded in the user's documents
        instead of failing the request. Score is the fraction of query terms a
        chunk contains, so it stays in [0, 1] like cosine similarity.
        """
        terms = [t for t in dict.fromkeys(t.strip() for t in terms) if t][:8]
        if not terms or self.count() == 0:
            return []

        # Chroma's $contains is case-sensitive; cover the common casings and
        # do the real (case-folded) scoring below.
        variants = list(dict.fromkeys(v for t in terms for v in (t, t.capitalize(), t.upper())))
        clauses = [{"$contains": v} for v in variants]
        where_document = clauses[0] if len(clauses) == 1 else {"$or": clauses}

        found = self.collection.get(
            where=where or None,
            where_document=where_document,
            limit=scan_limit,
            include=["documents", "metadatas"],
        )
        hits: list[SearchHit] = []
        for cid, doc, meta in zip(
            found.get("ids") or [], found.get("documents") or [], found.get("metadatas") or []
        ):
            folded = (doc or "").casefold()
            matched = sum(1 for t in terms if t.casefold() in folded)
            if not matched:
                continue
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    text=doc or "",
                    score=matched / len(terms),
                    metadata=dict(meta or {}),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def _rows(result: dict, key: str) -> list:
    """First row of a Chroma query result.

    Chroma returns numpy arrays for embeddings/distances, so the usual
    ``result.get(key) or [[]]`` idiom raises "truth value of an array with more
    than one element is ambiguous". Check for None explicitly instead.
    """
    value = result.get(key)
    if value is None or len(value) == 0:
        return []
    first = value[0]
    return [] if first is None else list(first)


def _clean_metadata(meta: dict) -> dict:
    """Chroma only accepts scalar metadata values."""
    out = {}
    for key, value in meta.items():
        if value is None:
            continue
        out[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return out


class IndexManager:
    """Owns the lifecycle of IndexVersion rows and their Chroma collections."""

    def __init__(self, session: Session, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.chroma = get_chroma(self.settings)

    # --- lookup -----------------------------------------------------------
    def active_version(self, tenant_id: str) -> IndexVersion | None:
        return self.session.scalars(
            select(IndexVersion)
            .where(
                IndexVersion.tenant_id == tenant_id,
                IndexVersion.status == IndexStatus.ACTIVE,
            )
            .order_by(IndexVersion.activated_at.desc())
        ).first()

    def active_store(self, tenant_id: str) -> VectorStore | None:
        version = self.active_version(tenant_id)
        if version is None:
            return None
        return VectorStore(self._collection(version.collection_name))

    def store_for(self, version: IndexVersion) -> VectorStore:
        return VectorStore(self._collection(version.collection_name))

    def _collection(self, name: str) -> Collection:
        return self.chroma.get_or_create_collection(
            name=name, metadata=dict(_COLLECTION_METADATA)
        )

    # --- build / swap -----------------------------------------------------
    def begin_build(self, tenant_id: str, *, notes: str = "") -> IndexVersion:
        version = IndexVersion(
            id=new_id(),
            tenant_id=tenant_id,
            collection_name=self._collection_name(tenant_id),
            status=IndexStatus.BUILDING,
            embed_model=self.settings.EMBED_MODEL,
            notes=notes,
        )
        self.session.add(version)
        self.session.flush()
        self._collection(version.collection_name)  # materialise it
        log.info("index.build_started", version=version.id, collection=version.collection_name)
        return version

    def _collection_name(self, tenant_id: str) -> str:
        safe_tenant = "".join(c if c.isalnum() else "_" for c in tenant_id)[:32]
        return f"{self.settings.COLLECTION_NAME}__{safe_tenant}__{new_id()[:10]}"

    def smoke_check(self, version: IndexVersion, *, expected_chunks: int) -> tuple[bool, str]:
        """Gate the swap. A build that lost vectors must never go live.

        Deliberately strict: the failure mode this exists to prevent is exactly
        the old behaviour, where a half-finished rebuild replaced a good index.
        """
        store = self.store_for(version)
        actual = store.count()
        if expected_chunks == 0:
            return False, "refusing to activate an empty index"
        if actual != expected_chunks:
            return False, f"chunk count mismatch: expected {expected_chunks}, found {actual}"

        # A collection that cannot answer a query is not servable.
        try:
            probe = store.collection.get(limit=1, include=["embeddings"])
            vectors = probe.get("embeddings")
            if vectors is None or len(vectors) == 0:
                return False, "no embeddings retrievable from new collection"
            store.query(list(vectors[0]), k=1)
        except Exception as exc:
            return False, f"probe query failed: {exc}"

        return True, "ok"

    def activate(self, version: IndexVersion, *, expected_chunks: int) -> bool:
        """Flip ACTIVE to ``version`` if it passes smoke checks.

        On failure the previous version stays active and the new one is marked
        FAILED — the caller sees False and the corpus is still servable.
        """
        with stage("index.activate", **{"aka.index_version": version.id}):
            ok, detail = self.smoke_check(version, expected_chunks=expected_chunks)
            if not ok:
                self.reject(version, detail)
                return False

            previous = self.active_version(version.tenant_id)
            version.status = IndexStatus.ACTIVE
            version.activated_at = utcnow()
            version.chunk_count = expected_chunks
            if previous is not None and previous.id != version.id:
                previous.status = IndexStatus.RETIRED
            self.session.flush()

            index_swaps.labels(outcome="success").inc()
            log.info(
                "index.swapped",
                new_version=version.id,
                previous=previous.id if previous else None,
                chunks=expected_chunks,
            )
            return True

    def reject(self, version: IndexVersion, detail: str) -> None:
        """Mark a build that failed its smoke check. The collection is kept
        for inspection until the next prune."""
        version.status = IndexStatus.FAILED
        version.notes = f"{version.notes or ''}\nsmoke check failed: {detail}".strip()
        self.session.flush()
        index_swaps.labels(outcome="rejected").inc()
        log.error("index.swap_rejected", version=version.id, detail=detail)

    def abandon(self, version: IndexVersion, reason: str) -> None:
        version.status = IndexStatus.FAILED
        version.notes = f"{version.notes or ''}\n{reason}".strip()
        self.session.flush()
        self._drop_collection(version.collection_name)
        index_swaps.labels(outcome="abandoned").inc()

    def reap_stale_builds(self, tenant_id: str) -> int:
        """Fail BUILDING versions left behind by a crashed build.

        Only safe to call while holding the tenant's build slot (the worker
        never runs two reindex jobs for one tenant at once): any BUILDING row
        seen then cannot belong to a live build.
        """
        stale = list(
            self.session.scalars(
                select(IndexVersion).where(
                    IndexVersion.tenant_id == tenant_id,
                    IndexVersion.status == IndexStatus.BUILDING,
                )
            )
        )
        for version in stale:
            self.abandon(version, "build never finished (worker died?)")
        if stale:
            log.warning("index.stale_builds_reaped", tenant=tenant_id, count=len(stale))
        return len(stale)

    def retire_active(self, tenant_id: str, reason: str) -> IndexVersion | None:
        """Take the tenant's index out of service on purpose.

        Used when every document was deleted: an empty corpus is the correct
        state, and leaving the old index active would keep deleted documents
        retrievable. This is the only path that leaves a tenant without an
        ACTIVE version, and it is never taken because a build failed.
        """
        current = self.active_version(tenant_id)
        if current is None:
            return None
        current.status = IndexStatus.RETIRED
        current.notes = f"{current.notes or ''}\nretired: {reason}".strip()
        self.session.flush()
        log.warning("index.retired", version=current.id, reason=reason)
        return current

    def status(self, tenant_id: str) -> dict:
        versions = list(
            self.session.scalars(
                select(IndexVersion)
                .where(IndexVersion.tenant_id == tenant_id)
                .order_by(IndexVersion.created_at.desc())
                .limit(10)
            )
        )
        active = next((v for v in versions if v.status is IndexStatus.ACTIVE), None)
        return {
            "active": _version_dict(active) if active else None,
            "building": [_version_dict(v) for v in versions if v.status is IndexStatus.BUILDING],
            "recent": [_version_dict(v) for v in versions],
        }

    def prune_retired(self, tenant_id: str, *, keep: int = 1) -> int:
        """Drop old collections, keeping ``keep`` retired versions for rollback."""
        retired = list(
            self.session.scalars(
                select(IndexVersion)
                .where(
                    IndexVersion.tenant_id == tenant_id,
                    IndexVersion.status.in_([IndexStatus.RETIRED, IndexStatus.FAILED]),
                )
                .order_by(IndexVersion.created_at.desc())
            )
        )
        dropped = 0
        for version in retired[keep:]:
            self._drop_collection(version.collection_name)
            self.session.delete(version)
            dropped += 1
        self.session.flush()
        return dropped

    def rollback_to_previous(self, tenant_id: str) -> IndexVersion | None:
        """Manual escape hatch: re-activate the most recent retired version."""
        current = self.active_version(tenant_id)
        candidate = self.session.scalars(
            select(IndexVersion)
            .where(
                IndexVersion.tenant_id == tenant_id,
                IndexVersion.status == IndexStatus.RETIRED,
            )
            .order_by(IndexVersion.activated_at.desc())
        ).first()
        if candidate is None:
            return None
        if current is not None:
            current.status = IndexStatus.RETIRED
        candidate.status = IndexStatus.ACTIVE
        candidate.activated_at = utcnow()
        self.session.flush()
        log.warning("index.rolled_back", to_version=candidate.id)
        return candidate

    def _drop_collection(self, name: str) -> None:
        try:
            self.chroma.delete_collection(name)
        except Exception as exc:  # already gone, or store unreachable
            log.debug("index.drop_failed", collection=name, error=str(exc))


def _version_dict(version: IndexVersion) -> dict:
    return {
        "id": version.id,
        "status": version.status.value,
        "collection": version.collection_name,
        "doc_count": version.doc_count,
        "chunk_count": version.chunk_count,
        "embed_model": version.embed_model,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "activated_at": version.activated_at.isoformat() if version.activated_at else None,
        "notes": version.notes,
    }
