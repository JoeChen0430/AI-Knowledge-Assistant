"""Request-scoped dependencies: settings, DB session, authenticated principal."""

from __future__ import annotations

from typing import Callable, Iterator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from aka.api.services import AppServices
from aka.db.session import get_session_factory
from aka.obs.logging import bind_request_context
from aka.security.auth import ROLE_ADMIN, AuthError, Principal, decode_token
from aka.settings import Settings

_bearer = HTTPBearer(auto_error=False)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def get_session(request: Request) -> Iterator[Session]:
    """One session per request. Routes commit explicitly before returning;
    this only guarantees rollback and close on every path."""
    session = get_session_factory(request.app.state.settings)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_principal(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """The caller, from a verified token. The tenant id always comes from
    here, never from a request parameter."""
    settings: Settings = request.app.state.settings
    if not settings.AUTH_ENABLED:
        principal = Principal(
            user_id=settings.DEFAULT_USER_ID,
            tenant_id=settings.DEFAULT_TENANT,
            email="local@localhost",
            role=ROLE_ADMIN,
        )
    else:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            principal = decode_token(creds.credentials, settings)
        except AuthError as exc:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}
            ) from exc

    bind_request_context(tenant_id=principal.tenant_id, user_id=principal.user_id)
    return principal


def require_role(role: str) -> Callable[..., Principal]:
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_role(role):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{role}' required")
        return principal

    return dependency
