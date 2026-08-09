"""FastAPI application factory."""
from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response

from app.api.routes.main_routes import router
from app.api.routes.agentic_routes import router as agentic_router
from app.core.config import get_settings
from app.observability.logging_setup import configure_logging
from app.observability import metrics


def create_app() -> FastAPI:
    configure_logging()
    s = get_settings()
    app = FastAPI(title=s.service_name, version=s.build_version)
    app.include_router(router)
    app.include_router(agentic_router)

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
