"""Rate limiting (token bucket per principal/IP) and idempotency helpers.

Starter uses in-process state; production should back these with Redis so limits
and idempotency keys are shared across replicas.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

_BUCKETS: dict[str, list[float]] = defaultdict(list)


def _client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth:
        return auth[-16:]  # coarse per-token key without logging the token
    return request.client.host if request.client else "anon"


async def rate_limit(request: Request) -> None:
    s = get_settings()
    key = _client_key(request)
    now = time.time()
    window_start = now - 60
    bucket = [t for t in _BUCKETS[key] if t >= window_start]
    if len(bucket) >= s.rate_limit_per_minute:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    bucket.append(now)
    _BUCKETS[key] = bucket


# Idempotency: map key -> cached run_id. Replace with Redis + TTL in prod.
_IDEMPOTENCY: dict[str, str] = {}


def idempotency_get(key: str | None) -> str | None:
    return _IDEMPOTENCY.get(key) if key else None


def idempotency_put(key: str | None, run_id: str) -> None:
    if key:
        _IDEMPOTENCY[key] = run_id
