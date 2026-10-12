"""FastAPI application.

    uvicorn aka.api.main:app --host 127.0.0.1 --port 8080

``create_app`` takes the settings and, for tests and evals, a model client to
use instead of Ollama; everything else is wired from settings.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aka.api.routes import auth, chat, documents, health, review
from aka.api.services import AppServices
from aka.db.session import init_db
from aka.hitl.approvals import ApprovalError
from aka.ingest.documents import DocumentError
from aka.obs.logging import clear_request_context, get_logger, setup_logging
from aka.obs.telemetry import current_trace_id, setup_telemetry
from aka.security.auth import AuthError
from aka.security.paths import UnsafePathError
from aka.settings import Settings, get_settings

log = get_logger("aka.api")


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: Any = None,
    start_worker: bool | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    services = AppServices.build(settings, llm_client=llm_client)
    run_worker = settings.EMBEDDED_WORKER if start_worker is None else start_worker

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings)
        settings.ensure_dirs()
        init_db(settings)
        if run_worker:
            services.start_worker()
        log.info(
            "api.started",
            env=settings.ENV,
            auth=settings.AUTH_ENABLED,
            embedded_worker=run_worker,
            fault_rate=settings.FAULT_INJECTION_RATE,
        )
        try:
            yield
        finally:
            services.stop_worker()
            log.info("api.stopped")

    app = FastAPI(
        title="AI Knowledge Assistant",
        version=settings.SERVICE_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = services

    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        trace_id = current_trace_id()
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        if request.url.path not in ("/healthz", "/metrics"):
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        return response

    @app.exception_handler(AuthError)
    async def _auth_error(_request: Request, exc: AuthError):
        code = 403 if exc.reason == "insufficient_role" else 401
        return JSONResponse({"detail": str(exc)}, status_code=code)

    @app.exception_handler(ApprovalError)
    async def _approval_error(_request: Request, exc: ApprovalError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(UnsafePathError)
    async def _unsafe_path(_request: Request, exc: UnsafePathError):
        return JSONResponse({"detail": "path not allowed"}, status_code=400)

    @app.exception_handler(DocumentError)
    async def _document_error(_request: Request, exc: DocumentError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    for module in (health, auth, chat, documents, review):
        app.include_router(module.router)

    if settings.OTEL_ENABLED:
        setup_telemetry(settings)
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")

    return app


def __getattr__(name: str):
    # ``uvicorn aka.api.main:app`` resolves this lazily, so importing the
    # module (tests, tooling) does not build an app from ambient settings.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
