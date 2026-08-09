"""API request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel

from app.domain.models import (
    NotificationRecord,
    RootCauseReport,
    TriageRecord,
)


class TokenRequest(BaseModel):
    username: str
    role: str = "analyst"
    tenant_id: str = "default"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AnalyzeResponse(BaseModel):
    run_id: str
    triage: list[TriageRecord]
    notifications: list[NotificationRecord]
    report: RootCauseReport


class HealthResponse(BaseModel):
    status: str
    provider_healthy: bool
    provider: str
    model: str
