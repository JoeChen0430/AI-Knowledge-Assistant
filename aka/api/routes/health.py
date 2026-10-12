"""Liveness, readiness, metrics and dependency health."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from aka.api.deps import get_services, get_settings_dep, require_role
from aka.api.services import AppServices
from aka.db.session import healthcheck
from aka.ingest.vectorstore import get_chroma
from aka.llm.breaker import all_breakers
from aka.obs import metrics
from aka.security.auth import ROLE_REVIEWER
from aka.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    """Liveness: the process is up. Deliberately checks nothing else, so a
    model outage never gets the API container restarted."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(
    response: Response,
    settings: Settings = Depends(get_settings_dep),
    services: AppServices = Depends(get_services),
) -> dict:
    """Readiness. Only the database is required: without the model backend
    the API still serves degraded answers, which is the point."""
    checks = {"database": healthcheck(settings)}
    try:
        get_chroma(settings).heartbeat()
        checks["vector_store"] = True
    except Exception:
        checks["vector_store"] = False
    checks["model_backend"] = bool(services.client.ping())

    ready = checks["database"]
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "checks": checks}


@router.get("/metrics")
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@router.get("/health/llm", dependencies=[Depends(require_role(ROLE_REVIEWER))])
def llm_health(services: AppServices = Depends(get_services)) -> dict:
    """Model chain, reachability, and every circuit breaker (LLM tiers and the
    embedding model)."""
    return {**services.router.health(), "breakers": all_breakers()}
