"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response

from app.api.routes.main_routes import router
from app.api.routes.agentic_routes import router as agentic_router
from app.api.routes.rootcause_routes import router as rootcause_router
from app.core.config import get_settings
from app.observability.logging_setup import configure_logging, log_event
from app.observability import metrics
from app.triggers.automation import watch_inbox

logger = logging.getLogger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the background file-drop watcher for the lifetime of the process.

    The watcher only starts when TRIGGER_INBOX_DIR is set, so tests, CLI runs and
    plain API usage never spawn a loop that scans the filesystem. On shutdown the
    task is cancelled and awaited so an in-flight batch is not abandoned silently.
    """
    s = get_settings()
    task: asyncio.Task | None = None
    if s.trigger_inbox_dir:
        inbox = Path(s.trigger_inbox_dir)
        archive = Path(s.trigger_archive_dir) if s.trigger_archive_dir else inbox.parent / "archive"
        task = asyncio.create_task(
            watch_inbox(s, inbox, archive), name="file-drop-watcher"
        )
    else:
        log_event(logger, "file_watch_disabled",
                  reason="TRIGGER_INBOX_DIR not set")
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def create_app() -> FastAPI:
    configure_logging()
    s = get_settings()
    app = FastAPI(title=s.service_name, version=s.build_version, lifespan=lifespan)
    app.include_router(router)
    app.include_router(agentic_router)
    app.include_router(rootcause_router)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - start
        metrics.API_LATENCY.labels(request.url.path, request.method).observe(elapsed)
        response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"
        return response

    @app.get("/metrics")
    async def prometheus_metrics():
        if not metrics.enabled():
            return Response("metrics disabled", media_type="text/plain")
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()