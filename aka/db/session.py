"""Engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from aka.db.models import Base
from aka.settings import Settings, get_settings

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _configure_sqlite(engine: Engine) -> None:
    """WAL + busy timeout so the API and the worker can share one file.

    Without WAL, a worker holding a write lock blocks every API read.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # pragma: no cover - driver callback
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine

    settings = settings or get_settings()
    url = settings.DATABASE_URL
    kwargs: dict = {"pool_pre_ping": True, "future": True}

    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)

    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        _configure_sqlite(_engine)

    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    if _SessionFactory is None:
        get_engine(settings)
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_INIT_LOCK_KEY = 0x414B41  # arbitrary, stable advisory-lock id ("AKA")


def init_db(settings: Settings | None = None) -> None:
    """Create tables. Fine for a single-node deploy; swap for Alembic when
    more than one environment needs to migrate independently.

    On Postgres the API and the worker start together and would race
    ``CREATE TABLE``; an advisory lock serialises them.
    """
    engine = get_engine(settings)
    if engine.dialect.name != "postgresql":
        Base.metadata.create_all(engine)
        return
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _INIT_LOCK_KEY})
        Base.metadata.create_all(conn)


def healthcheck(settings: Settings | None = None) -> bool:
    try:
        with get_engine(settings).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_engine_for_tests() -> None:
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
