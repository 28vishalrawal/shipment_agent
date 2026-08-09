"""OAuth2/JWT auth with role-based scopes. Roles map to coarse capabilities."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


class Role(str, Enum):
    ADMIN = "admin"
    OPERATIONS_MANAGER = "operations_manager"
    SUPPORT_AGENT = "support_agent"
    ANALYST = "analyst"
    VIEWER = "viewer"


# Capability scopes per role.
ROLE_SCOPES: dict[Role, set[str]] = {
    Role.ADMIN: {"triage:run", "triage:read", "notify:send", "analytics:run", "analytics:read", "admin"},
    Role.OPERATIONS_MANAGER: {"triage:run", "triage:read", "notify:send", "analytics:run", "analytics:read"},
    Role.SUPPORT_AGENT: {"triage:read", "notify:send"},
    Role.ANALYST: {"analytics:run", "analytics:read", "triage:read"},
    Role.VIEWER: {"triage:read", "analytics:read"},
}


def create_token(subject: str, role: Role, tenant_id: str = "default") -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role.value,
        "tenant_id": tenant_id,
        "scopes": sorted(ROLE_SCOPES[role]),
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_expiry_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


class Principal:
    def __init__(self, sub: str, role: str, tenant_id: str, scopes: set[str]) -> None:
        self.sub = sub
        self.role = role
        self.tenant_id = tenant_id
        self.scopes = scopes


async def get_principal(token: str | None = Depends(oauth2_scheme)) -> Principal:
    s = get_settings()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing token")
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    return Principal(payload["sub"], payload["role"], payload.get("tenant_id", "default"),
                     set(payload.get("scopes", [])))


def require_scope(scope: str):
    async def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if scope not in principal.scopes and "admin" not in principal.scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"missing scope {scope}")
        return principal
    return _dep
