"""Shared fixtures. Every test runs against a throwaway SQLite file and a
temp DATA_DIR so nothing touches the developer's real corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Must be set before aka.settings is imported anywhere.
os.environ.setdefault("AKA_EAGER_SETTINGS", "0")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("OTEL_ENABLED", "false")
os.environ.setdefault("AUTH_ENABLED", "true")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-0123456789")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """Isolated Settings pointing at tmp_path for all mutable state."""
    from aka.settings import Settings, reset_settings_cache

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRASH_DIR", str(tmp_path / "trash"))
    monkeypatch.setenv("QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setenv("DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("OTEL_ENABLED", "false")
    reset_settings_cache()

    s = Settings()
    s.ensure_dirs()
    Path(s.DB_DIR).mkdir(parents=True, exist_ok=True)
    yield s
    reset_settings_cache()


@pytest.fixture()
def db(settings):
    """Initialised database bound to the isolated settings."""
    from aka.db.session import get_session_factory, init_db, reset_engine_for_tests

    reset_engine_for_tests()
    init_db(settings)
    factory = get_session_factory(settings)
    yield factory
    reset_engine_for_tests()


@pytest.fixture()
def session(db):
    s = db()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Clear process-global registries so tests cannot leak state into each
    other. The breaker registry in particular is keyed by model name, which
    several tests reuse."""
    from aka.llm.breaker import clear_breaker_registry
    from aka.llm.cache import reset_answer_cache
    from aka.llm.router import reset_router

    clear_breaker_registry()
    reset_answer_cache()
    reset_router()
    yield
    clear_breaker_registry()
    reset_answer_cache()
    reset_router()
