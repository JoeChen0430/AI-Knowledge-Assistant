"""Document operations behind the API: upload, listing, soft delete, undo.

Every path that turns user input into a filesystem location goes through
:func:`aka.security.paths.safe_join`. Deletes are moves into the trash with an
undo token, and every change to the corpus queues a reindex instead of
rebuilding inline, so a request never waits on embedding.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.db.models import Document, DocumentStatus, Job
from aka.hitl import approvals
from aka.hitl.policy import Action, PolicyEngine, Proposal, Verdict
from aka.ingest import queue
from aka.ingest.pipeline import scan_data_dir
from aka.obs.logging import get_logger
from aka.security.paths import prune_empty_dirs, relative_label, safe_join, sanitize_filename
from aka.settings import Settings

log = get_logger("aka.documents")


class DocumentError(ValueError):
    """A request about a document that cannot be honoured (bad input, not found)."""


@dataclass
class StoredFile:
    rel_path: str
    size_bytes: int


def store_upload(settings: Settings, *, filename: str, data: bytes, subfolder: str = "") -> StoredFile:
    """Validate and write one uploaded file under DATA_DIR."""
    safe_name = sanitize_filename(filename)
    if Path(safe_name).suffix.lower() not in settings.ALLOWED_EXTENSIONS:
        raise DocumentError(f"unsupported file type: {Path(safe_name).suffix or '(none)'}")
    if not data:
        raise DocumentError("file is empty")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise DocumentError(f"file exceeds {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    # Raises UnsafePathError for "../" and absolute paths.
    target_dir = safe_join(settings.DATA_DIR, subfolder or "")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = safe_join(target_dir, safe_name)

    # Write to a temp file and rename, so the ingest scan never reads a
    # half-written upload.
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".upload-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    rel = relative_label(dest, settings.DATA_DIR)
    log.info("document.uploaded", path=rel, bytes=len(data))
    return StoredFile(rel_path=rel, size_bytes=len(data))


def list_documents(session: Session, settings: Settings, *, tenant_id: str) -> list[dict]:
    """Indexed documents plus files on disk the index has not seen yet."""
    rows = {
        d.rel_path: d
        for d in session.scalars(
            select(Document).where(
                Document.tenant_id == tenant_id, Document.status != DocumentStatus.DELETED
            )
        )
    }
    on_disk = scan_data_dir(settings)
    out = []
    for rel_path in sorted(set(rows) | set(on_disk)):
        doc = rows.get(rel_path)
        out.append(
            {
                "id": doc.id if doc else None,
                "rel_path": rel_path,
                "status": doc.status.value if doc else "pending",
                "on_disk": rel_path in on_disk,
                "size_bytes": doc.size_bytes if doc else on_disk[rel_path].stat().st_size,
                "updated_at": doc.updated_at.isoformat() if doc and doc.updated_at else None,
            }
        )
    return out


def soft_delete(
    session: Session,
    settings: Settings,
    *,
    tenant_id: str,
    document_id: str,
    actor_id: str,
) -> tuple[dict, Job]:
    """Move a document to the trash, issue an undo token, queue a reindex."""
    doc = session.scalars(
        select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    ).first()
    if doc is None or doc.status is DocumentStatus.DELETED:
        raise DocumentError("document not found")

    source = safe_join(settings.DATA_DIR, doc.rel_path)
    if not source.exists():
        raise DocumentError("file is already gone from the data directory")

    proposal = Proposal(
        action=Action.DOCUMENT_DELETE,
        tenant_id=tenant_id,
        subject_type="document",
        subject_id=doc.id,
        title=doc.rel_path,
        actor_is_human=True,
        irreversible=False,
        payload={"rel_path": doc.rel_path},
    )
    decision = PolicyEngine(settings).evaluate(proposal)
    if decision.verdict is not Verdict.ALLOW:
        raise DocumentError(f"delete not allowed: {decision.reason}")
    approvals.record_auto_decision(session, proposal=proposal, decision=decision)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    trash_dest = safe_join(settings.TRASH_DIR, tenant_id, stamp, doc.rel_path)
    trash_dest.parent.mkdir(parents=True, exist_ok=True)
    # shutil.move, not Path.replace: in compose the data dir and the trash
    # can be different volumes, and rename() cannot cross devices.
    shutil.move(str(source), str(trash_dest))
    prune_empty_dirs(source.parent, Path(settings.DATA_DIR))

    doc.trash_path = str(trash_dest)
    token = approvals.issue_undo(
        session,
        tenant_id=tenant_id,
        action="document_delete",
        payload={"document_id": doc.id, "rel_path": doc.rel_path, "trash_path": str(trash_dest)},
        created_by=actor_id,
        settings=settings,
    )
    job = queue.ensure_reindex(session, tenant_id=tenant_id, reason="document deleted", settings=settings)
    log.info("document.deleted", path=doc.rel_path, trash=str(trash_dest))
    return (
        {
            "rel_path": doc.rel_path,
            "undo_token": token.id,
            "undo_expires_at": token.expires_at.isoformat(),
        },
        job,
    )


def undo_delete(session: Session, settings: Settings, *, tenant_id: str, token_id: str) -> tuple[dict, Job]:
    """Restore a trashed document. The token is consumed exactly once."""
    token = approvals.consume_undo(session, token_id=token_id, tenant_id=tenant_id)
    if token.action != "document_delete":
        raise DocumentError(f"undo token is for '{token.action}', not a document delete")

    trash_path = Path(token.payload["trash_path"])
    if not trash_path.exists():
        raise DocumentError("the trashed file is no longer available")
    dest = safe_join(settings.DATA_DIR, token.payload["rel_path"])
    if dest.exists():
        raise DocumentError("a file already exists at the original path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trash_path), str(dest))

    doc = session.get(Document, token.payload["document_id"])
    if doc is not None and doc.tenant_id == tenant_id:
        doc.trash_path = None
    job = queue.ensure_reindex(session, tenant_id=tenant_id, reason="delete undone", settings=settings)
    log.info("document.restored", path=token.payload["rel_path"])
    return {"rel_path": token.payload["rel_path"], "restored": True}, job
