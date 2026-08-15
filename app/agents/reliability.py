"""Retry with exponential backoff + a simple circuit breaker for LLM calls."""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from app.providers.base import ProviderError


@dataclass
class CircuitBreaker:
    fail_threshold: int = 5
    reset_after_s: float = 30.0
    _failures: int = 0
    _opened_at: float = 0.0

    def allow(self) -> bool:
        if self._failures < self.fail_threshold:
            return True
        if time.time() - self._opened_at > self.reset_after_s:
            self._failures = 0  # half-open: allow a trial
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold:
            self._opened_at = time.time()


# 4xx codes that signal a broken request or bad credentials rather than a
# transient fault. Retrying these burns the retry budget and trips the breaker
# for the rest of the run, turning a fixable config error (wrong key, wrong auth
# header, unknown model name) into a silent slide onto template fallbacks.
# 408 and 429 are excluded: those are genuinely worth retrying.
_NON_RETRYABLE = {400, 401, 403, 404, 405, 422}


def _is_retryable(exc: ProviderError) -> bool:
    return getattr(exc, "status_code", None) not in _NON_RETRYABLE


async def with_retries(coro_factory, *, max_retries: int, breaker: CircuitBreaker):
    """coro_factory: zero-arg callable returning a fresh awaitable each attempt."""
    if not breaker.allow():
        raise ProviderError("circuit_open")
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await coro_factory()
            breaker.record_success()
            return result, attempt
        except ProviderError as exc:
            last = exc
            if not _is_retryable(exc):
                # Surface immediately and leave the breaker untouched — the
                # endpoint is healthy, the request is wrong.
                raise
            breaker.record_failure()
            if attempt >= max_retries or not breaker.allow():
                break
            backoff = min(8.0, (2 ** attempt) * 0.5) + random.uniform(0, 0.25)
            await asyncio.sleep(backoff)
    raise last or ProviderError("unknown")