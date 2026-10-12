"""Authentication and tenant isolation.

Two rules this module exists to enforce:

1. A caller's ``tenant_id`` comes from the verified token, never from a request
   parameter. Client-supplied tenant ids are the classic multi-tenant data leak.
2. Every retrieval is filtered by that tenant server-side, so a bug in the UI
   cannot surface another tenant's documents.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from aka.db.models import User, new_id
from aka.obs.logging import get_logger
from aka.obs.metrics import auth_failures
from aka.settings import Settings, get_settings

log = get_logger("aka.auth")

ROLE_USER = "user"
ROLE_REVIEWER = "reviewer"
ROLE_ADMIN = "admin"
# Higher rank implies every capability of the ranks below it.
_ROLE_RANK = {ROLE_USER: 0, ROLE_REVIEWER: 1, ROLE_ADMIN: 2}

_PBKDF2_ROUNDS = 240_000


class AuthError(Exception):
    """Credential or token rejected."""

    def __init__(self, message: str, reason: str = "invalid"):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Everything downstream scopes off this."""

    user_id: str
    tenant_id: str
    email: str
    role: str = ROLE_USER

    def has_role(self, required: str) -> bool:
        return _ROLE_RANK.get(self.role, -1) >= _ROLE_RANK.get(required, 99)

    def require(self, required: str) -> None:
        if not self.has_role(required):
            raise AuthError(f"role '{required}' required", reason="insufficient_role")


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 with a per-password salt.

    Not argon2 only because it keeps the dependency set to the standard
    library; the iteration count is tuned to ~100ms.
    """
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        # Still burn time so a missing user is not distinguishable by timing.
        hashlib.pbkdf2_hmac("sha256", (password or "").encode(), b"decoy", _PBKDF2_ROUNDS)
        return False
    try:
        algo, rounds, salt_hex, digest_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

def issue_token(principal: Principal, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": principal.user_id,
        "tid": principal.tenant_id,
        "email": principal.email,
        "role": principal.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.JWT_TTL_S)).timestamp()),
        "iss": settings.SERVICE_NAME,
        "jti": new_id(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, settings: Settings | None = None) -> Principal:
    settings = settings or get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET,
            # Pin the algorithm: accepting the token's own 'alg' enables the
            # classic "alg: none" and HS/RS confusion attacks.
            algorithms=[settings.JWT_ALGORITHM],
            issuer=settings.SERVICE_NAME,
            options={"require": ["exp", "sub", "tid", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        auth_failures.labels(reason="expired").inc()
        raise AuthError("token expired", reason="expired") from exc
    except jwt.InvalidTokenError as exc:
        auth_failures.labels(reason="invalid_token").inc()
        raise AuthError("invalid token", reason="invalid_token") from exc

    return Principal(
        user_id=claims["sub"],
        tenant_id=claims["tid"],
        email=claims.get("email", ""),
        role=claims.get("role", ROLE_USER),
    )


# --------------------------------------------------------------------------
# User management
# --------------------------------------------------------------------------

def create_user(
    session: Session,
    *,
    tenant_id: str,
    email: str,
    password: str,
    role: str = ROLE_USER,
) -> User:
    if role not in _ROLE_RANK:
        raise ValueError(f"unknown role: {role}")
    user = User(
        id=new_id(),
        tenant_id=tenant_id,
        email=email.strip().lower(),
        password_hash=hash_password(password),
        role=role,
    )
    session.add(user)
    session.flush()
    log.info("auth.user_created", user_id=user.id, tenant=tenant_id, role=role)
    return user


def authenticate(session: Session, *, tenant_id: str, email: str, password: str) -> Principal:
    user = session.scalars(
        select(User).where(
            User.tenant_id == tenant_id, User.email == email.strip().lower()
        )
    ).first()

    if user is None or not user.is_active:
        # Run the verify anyway to keep the timing profile flat.
        verify_password(password, None)
        auth_failures.labels(reason="unknown_user").inc()
        raise AuthError("invalid credentials", reason="unknown_user")

    if not verify_password(password, user.password_hash):
        auth_failures.labels(reason="bad_password").inc()
        raise AuthError("invalid credentials", reason="bad_password")

    return Principal(
        user_id=user.id, tenant_id=user.tenant_id, email=user.email, role=user.role
    )


def principal_from_user(user: User) -> Principal:
    return Principal(
        user_id=user.id, tenant_id=user.tenant_id, email=user.email, role=user.role
    )
