"""Central configuration.

Every knob is env-overridable so the same image can run on a laptop
(SQLite + local Ollama, no Docker) or in compose (Postgres + collector).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
# Long enough for HS256 (>= 32 bytes) so local runs are quiet; refused in production.
DEV_JWT_SECRET = "dev-insecure-secret-change-me-before-deploying"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service identity -------------------------------------------------
    ENV: str = "local"
    SERVICE_NAME: str = "ai-knowledge-assistant"
    SERVICE_VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # --- storage ----------------------------------------------------------
    DATA_DIR: Path = REPO_ROOT / "data"
    TRASH_DIR: Path = REPO_ROOT / "var" / "trash"
    QUARANTINE_DIR: Path = REPO_ROOT / "var" / "quarantine"
    DB_DIR: Path = REPO_ROOT / "chroma_db"
    DATABASE_URL: str = f"sqlite:///{REPO_ROOT / 'var' / 'aka.db'}"
    COLLECTION_NAME: str = "personal_knowledge"
    CHROMA_HOST: str | None = None
    CHROMA_PORT: int = 8000

    # --- ingestion --------------------------------------------------------
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    ALLOWED_EXTENSIONS: Annotated[set[str], NoDecode] = {".pdf", ".txt", ".md", ".docx"}
    INGEST_MAX_ATTEMPTS: int = 3
    INGEST_BACKOFF_BASE_S: float = 2.0
    INGEST_POLL_INTERVAL_S: float = 1.0
    INGEST_LEASE_SECONDS: int = 600

    # --- models -----------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3"
    LLM_FALLBACK_MODELS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["llama3.2:1b"]
    )
    EMBED_MODEL: str = "nomic-embed-text"
    LLM_TIMEOUT_S: float = 90.0
    EMBED_TIMEOUT_S: float = 30.0
    LLM_MAX_TOKENS: int = 1024

    # --- resilience -------------------------------------------------------
    BREAKER_FAILURE_THRESHOLD: int = 5
    BREAKER_RECOVERY_TIMEOUT_S: float = 30.0
    BREAKER_HALF_OPEN_MAX_CALLS: int = 2
    REQUEST_BUDGET_S: float = 120.0
    ANSWER_CACHE_TTL_S: int = 900
    ANSWER_CACHE_MAX_ENTRIES: int = 512
    # Embedding is a single point of failure for retrieval, so it gets its own
    # retry budget and a lexical fallback instead of riding on the LLM chain.
    EMBED_RETRY_ATTEMPTS: int = 3
    EMBED_RETRY_BACKOFF_S: float = 0.25
    QUERY_EMBED_CACHE_SIZE: int = 1024
    LEXICAL_FALLBACK_ENABLED: bool = True

    # --- fault injection (chaos testing; off unless explicitly enabled) -----
    FAULT_INJECTION_RATE: float = 0.0
    FAULT_INJECTION_SEED: int | None = None
    FAULT_INJECTION_TARGETS: Annotated[set[str], NoDecode] = {"chat", "embed"}
    # Simulated timeouts sleep this long before raising, so the time budget is
    # exercised rather than every fault being an instant failure.
    FAULT_INJECTION_TIMEOUT_DELAY_S: float = 0.5

    # --- retrieval --------------------------------------------------------
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_RERANK_TO: int = 4
    RETRIEVAL_MIN_SCORE: float = 0.25

    # --- tiered memory ----------------------------------------------------
    L0_MAX_TURNS: int = 6
    L0_TOKEN_BUDGET: int = 1500
    L1_TTL_S: int = 60 * 60 * 24 * 7
    L2_TOP_K: int = 5
    L2_HALF_LIFE_DAYS: float = 30.0
    L2_MIN_RETRIEVAL_SCORE: float = 0.35
    MEMORY_AUTO_COMMIT_CONFIDENCE: float = 0.80
    MEMORY_REJECT_CONFIDENCE: float = 0.35
    # Post-turn L1 summarisation + L2 fact extraction. Costs one or two extra
    # model calls per turn, off the request path.
    MEMORY_EXTRACTION_ENABLED: bool = True

    # --- human-in-the-loop ------------------------------------------------
    HITL_ENABLED: bool = True
    # Answers below MIN are refused outright; between MIN and REVIEW they are
    # shown, flagged, and queued for a human; at or above REVIEW they release.
    ANSWER_MIN_GROUNDING: float = 0.15
    ANSWER_REVIEW_GROUNDING: float = 0.40
    APPROVAL_TTL_HOURS: int = 72
    UNDO_WINDOW_HOURS: int = 24

    # --- security ---------------------------------------------------------
    AUTH_ENABLED: bool = True
    JWT_SECRET: str = DEV_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    JWT_TTL_S: int = 60 * 60 * 12
    BOOTSTRAP_ADMIN_TOKEN: str | None = None
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_BURST: int = 10
    REDIS_URL: str | None = None

    # --- observability ----------------------------------------------------
    OTEL_ENABLED: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_CONSOLE_EXPORT: bool = False
    OTEL_SAMPLE_RATIO: float = 1.0
    METRICS_ENABLED: bool = True

    # --- api --------------------------------------------------------------
    API_BASE_URL: str = "http://localhost:8080"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8080
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # With auth off, every request runs as this principal. Meant for the
    # single-user laptop setup; production refuses to start that way.
    DEFAULT_TENANT: str = "default"
    DEFAULT_USER_ID: str = "local"
    # Run the ingest worker as a thread inside the API process. Right for a
    # laptop (one process, embedded Chroma); compose runs a separate worker.
    EMBEDDED_WORKER: bool = True

    @field_validator("LLM_FALLBACK_MODELS", "CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _split_ext(cls, v):
        if isinstance(v, str):
            return {e.strip().lower() for e in v.split(",") if e.strip()}
        return v

    @field_validator("FAULT_INJECTION_TARGETS", mode="before")
    @classmethod
    def _split_targets(cls, v):
        if isinstance(v, str):
            return {t.strip().lower() for t in v.split(",") if t.strip()}
        return v

    @field_validator("DATA_DIR", "TRASH_DIR", "QUARANTINE_DIR", "DB_DIR", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    def ensure_dirs(self) -> None:
        for p in (self.DATA_DIR, self.TRASH_DIR, self.QUARANTINE_DIR, REPO_ROOT / "var"):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() in {"prod", "production", "staging"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if s.is_production and s.JWT_SECRET == DEV_JWT_SECRET:
        raise RuntimeError("JWT_SECRET must be set outside local development")
    if s.is_production and not s.AUTH_ENABLED:
        raise RuntimeError("AUTH_ENABLED=false is only allowed outside production")
    if s.is_production and s.FAULT_INJECTION_RATE > 0:
        raise RuntimeError("fault injection must not be enabled in production")
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    """Test hook: forget the memoised Settings instance."""
    get_settings.cache_clear()


settings = get_settings() if os.getenv("AKA_EAGER_SETTINGS", "1") == "1" else None
