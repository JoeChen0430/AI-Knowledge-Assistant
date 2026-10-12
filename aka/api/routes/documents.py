"""Corpus management: list, upload, delete with undo, index status and rebuild."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.api.deps import get_principal, get_session, get_settings_dep, require_role
from aka.api.schemas import RebuildRequest
from aka.db.models import Job
from aka.hitl.approvals import ApprovalError
from aka.ingest import documents, queue
from aka.ingest.vectorstore import IndexManager
from aka.security.auth import ROLE_ADMIN, Principal
from aka.security.paths import UnsafePathError
from aka.settings import Settings

router = APIRouter(tags=["documents"])


def _job_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status.value,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "last_error": job.last_error,
        "result": job.result or {},
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@router.get("/documents")
def list_documents(
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> list[dict]:
    return documents.list_documents(session, settings, tenant_id=principal.tenant_id)


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED)
def upload_documents(
    files: list[UploadFile] = File(...),
    subfolder: str = Form(default=""),
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Store files and queue a reindex. Returns immediately: embedding runs
    in the worker and the current index keeps serving until the new one is
    swapped in."""
    stored, rejected = [], []
    for upload in files:
        data = upload.file.read(settings.MAX_UPLOAD_BYTES + 1)
        try:
            saved = documents.store_upload(
                settings, filename=upload.filename or "upload", data=data, subfolder=subfolder
            )
            stored.append({"rel_path": saved.rel_path, "size_bytes": saved.size_bytes})
        except UnsafePathError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "that subfolder name is not allowed") from exc
        except documents.DocumentError as exc:
            rejected.append({"filename": upload.filename, "reason": str(exc)})

    job = None
    if stored:
        job = queue.ensure_reindex(
            session, tenant_id=principal.tenant_id, reason="upload", settings=settings
        )
        session.commit()
    return {"stored": stored, "rejected": rejected, "job": _job_dict(job) if job else None}


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    try:
        info, job = documents.soft_delete(
            session,
            settings,
            tenant_id=principal.tenant_id,
            document_id=document_id,
            actor_id=principal.user_id,
        )
    except documents.DocumentError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    session.commit()
    return {**info, "job": _job_dict(job)}


@router.post("/documents/undo/{token_id}")
def undo_delete(
    token_id: str,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    try:
        info, job = documents.undo_delete(
            session, settings, tenant_id=principal.tenant_id, token_id=token_id
        )
    except (ApprovalError, documents.DocumentError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    session.commit()
    return {**info, "job": _job_dict(job)}


@router.get("/index/status")
def index_status(
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    jobs = session.scalars(
        select(Job)
        .where(Job.tenant_id == principal.tenant_id)
        .order_by(Job.created_at.desc())
        .limit(10)
    )
    return {
        **IndexManager(session, settings).status(principal.tenant_id),
        "jobs": [_job_dict(j) for j in jobs],
    }


@router.post("/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
def rebuild_index(
    body: RebuildRequest,
    principal: Principal = Depends(require_role(ROLE_ADMIN)),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    if body.force_full:
        job = queue.enqueue(
            session,
            tenant_id=principal.tenant_id,
            kind="reindex",
            payload={"force_full": True, "reason": "manual full rebuild"},
            settings=settings,
        )
    else:
        job = queue.ensure_reindex(
            session, tenant_id=principal.tenant_id, reason="manual", settings=settings
        )
    session.commit()
    return _job_dict(job)


@router.post("/index/rollback")
def rollback_index(
    principal: Principal = Depends(require_role(ROLE_ADMIN)),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """Re-activate the previous index version (kept by the pruner for this).

    An escape hatch for a bad swap: the documents table still describes the
    newer corpus, so run a ``force_full`` rebuild once the cause is fixed.
    """
    version = IndexManager(session, settings).rollback_to_previous(principal.tenant_id)
    if version is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "no previous index version to roll back to")
    session.commit()
    return {"active_version": version.id}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(get_principal),
    session: Session = Depends(get_session),
) -> dict:
    job = session.get(Job, job_id)
    if job is None or job.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return _job_dict(job)
