"""Prometheus metrics. Import-safe even if prometheus_client is absent."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge

    API_LATENCY = Histogram("api_latency_seconds", "API latency", ["route", "method"])
    LLM_LATENCY = Histogram("llm_latency_seconds", "LLM call latency", ["provider", "model"])
    LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM cost", ["provider", "model"])
    GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Guardrail blocks", ["stage", "reason"])
    EXCEPTIONS_DETECTED = Counter("exceptions_detected_total", "At-risk shipments flagged")
    ESCALATIONS = Counter("escalations_generated_total", "Systemic escalations generated")
    QUEUE_DEPTH = Gauge("job_queue_depth", "Pending async jobs")
    BATCH_DURATION = Histogram("batch_duration_seconds", "Analytics batch duration")
    ERRORS = Counter("errors_total", "Errors", ["type"])
    _ENABLED = True
except Exception:  # pragma: no cover
    _ENABLED = False

    class _Noop:
        def labels(self, *a, **k):
            return self
        def observe(self, *a, **k):
            return None
        def inc(self, *a, **k):
            return None
        def set(self, *a, **k):
            return None

    API_LATENCY = LLM_LATENCY = LLM_COST = GUARDRAIL_BLOCKS = _Noop()
    EXCEPTIONS_DETECTED = ESCALATIONS = QUEUE_DEPTH = BATCH_DURATION = ERRORS = _Noop()


def enabled() -> bool:
    return _ENABLED
