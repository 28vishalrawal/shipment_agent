"""Primary API routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile

from app.analytics.ingest import ingest
from app.api.middleware import idempotency_get, idempotency_put, rate_limit
from app.api.schemas import (
    AnalyzeResponse,
    HealthResponse,
    TokenRequest,
    TokenResponse,
)
from app.api.security import Principal, Role, create_token, require_scope
from app.api.upload_validation import read_upload
from app.agents.orchestrator import Orchestrator
from app.core.config import get_settings
from app.observability.logging_setup import log_event, new_id
from app.providers.factory import build_provider

logger = logging.getLogger("api")
router = APIRouter()

# Response cache keyed by run_id, paired with the idempotency-key map in
# middleware. Both are in-memory in the starter; back with Redis in production.
_RUN_CACHE: dict[str, "AnalyzeResponse"] = {}


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(req: TokenRequest) -> TokenResponse:
    # Assumption: identity is provided by an upstream IdP in prod. This dev
    # endpoint mints a scoped token for a stated role for local testing only.
    role = Role(req.role)
    return TokenResponse(access_token=create_token(req.username, role, req.tenant_id))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    provider = build_provider(s)
    healthy = await provider.health_check()
    return HealthResponse(status="ok", provider_healthy=healthy,
                          provider=provider.name, model=provider.model)


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    principal: Principal = Depends(require_scope("analytics:run")),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    _rl: None = Depends(rate_limit),
) -> AnalyzeResponse:
    correlation_id = new_id()
    log_event(logger, "request_received", correlation_id=correlation_id,
              route="/v1/analyze", tenant_id=principal.tenant_id, role=principal.role)

    # Idempotency: a repeated key returns the cached run instead of recomputing.
    cached_run = idempotency_get(idempotency_key)
    if cached_run and cached_run in _RUN_CACHE:
        log_event(logger, "request_received", status="idempotent_replay",
                  correlation_id=correlation_id, run_id=cached_run)
        return _RUN_CACHE[cached_run]

    df = await read_upload(file)
    ingested = ingest(df)

    s = get_settings()
    provider = build_provider(s)
    orch = Orchestrator(provider, s)
    triage, notifications, report = await orch.run(ingested, correlation_id)

    response = AnalyzeResponse(
        run_id=report.run_id, triage=triage, notifications=notifications, report=report
    )
    idempotency_put(idempotency_key, report.run_id)
    _RUN_CACHE[report.run_id] = response
    return response
