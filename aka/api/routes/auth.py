"""Login, first-admin bootstrap, and user management."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aka.api.deps import get_principal, get_session, get_settings_dep, require_role
from aka.api.schemas import BootstrapRequest, CreateUserRequest, LoginRequest
from aka.db.models import User
from aka.security.auth import (
    ROLE_ADMIN,
    AuthError,
    Principal,
    authenticate,
    create_user,
    issue_token,
    principal_from_user,
)
from aka.settings import Settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
def me(
    principal: Principal = Depends(get_principal),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    return {
        "user_id": principal.user_id,
        "tenant_id": principal.tenant_id,
        "email": principal.email,
        "role": principal.role,
        "auth_enabled": settings.AUTH_ENABLED,
    }


@router.post("/login")
def login(
    body: LoginRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    try:
        principal = authenticate(
            session, tenant_id=body.tenant_id, email=body.email, password=body.password
        )
    except AuthError as exc:
        # One message for unknown user and bad password: no account probing.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials") from exc
    return {
        "access_token": issue_token(principal, settings),
        "token_type": "bearer",
        "expires_in": settings.JWT_TTL_S,
        "role": principal.role,
    }


@router.post("/bootstrap", status_code=status.HTTP_201_CREATED)
def bootstrap(
    body: BootstrapRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings_dep),
    x_bootstrap_token: str | None = Header(default=None),
) -> dict:
    """Create the first admin of a tenant. Needs BOOTSTRAP_ADMIN_TOKEN and
    only works while the tenant has no users."""
    expected = settings.BOOTSTRAP_ADMIN_TOKEN
    if not expected or not x_bootstrap_token or not hmac.compare_digest(expected, x_bootstrap_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bootstrap is disabled or the token is wrong")
    existing = session.scalar(
        select(func.count(User.id)).where(User.tenant_id == body.tenant_id)
    )
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "tenant already has users")
    user = create_user(
        session, tenant_id=body.tenant_id, email=body.email, password=body.password, role=ROLE_ADMIN
    )
    session.commit()
    principal = principal_from_user(user)
    return {"user_id": user.id, "access_token": issue_token(principal, settings), "token_type": "bearer"}


@router.post("/users", status_code=status.HTTP_201_CREATED)
def add_user(
    body: CreateUserRequest,
    principal: Principal = Depends(require_role(ROLE_ADMIN)),
    session: Session = Depends(get_session),
) -> dict:
    """Admins add users to their own tenant only."""
    try:
        user = create_user(
            session,
            tenant_id=principal.tenant_id,
            email=body.email,
            password=body.password,
            role=body.role,
        )
        session.commit()
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "a user with that email already exists") from exc
    return {"user_id": user.id, "email": user.email, "role": user.role}
