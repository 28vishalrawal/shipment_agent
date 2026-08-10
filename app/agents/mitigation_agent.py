"""Lane B mitigation agent. LLM explains validated metrics only; computes nothing."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from app.agents.reliability import CircuitBreaker, with_retries
from app.core.config import Settings
from app.domain.models import ValidatedFinding
from app.observability.logging_setup import log_event
from app.observability import metrics
from app.prompts import mitigation_v1
from app.providers.base import LLMProvider, ProviderError

logger = logging.getLogger("agent.mitigation")


class _MitigationSchema(BaseModel):
    narrative: str
    mitigation: str
    expected_effect: str


class MitigationAgent:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._breaker = CircuitBreaker()

    async def explain(self, finding: ValidatedFinding, correlation_id: str) -> ValidatedFinding:
        # Classify the finding's shape so the LLM can match a mitigation lever
        # instead of defaulting to one. Deterministic hint, not a computed metric.
        dim_names = set(finding.dims.keys())
        if dim_names == {"shipping_mode"}:
            shape = "single_mode"
        elif "shipping_mode" in dim_names and dim_names & {"order_region", "market"}:
            shape = "mode_and_lane"
        elif dim_names & {"category", "department"}:
            shape = "product_category"
        elif dim_names == {"customer_segment"}:
            shape = "customer_segment"
        elif "shipping_mode" in dim_names:
            shape = "mode_combination"
        else:
            shape = "other"

        facts = {
            "segment": finding.label,
            "dimensions": ", ".join(f"{k}={v}" for k, v in finding.dims.items()),
            "finding_shape": shape,
            "n_orders": finding.n,
            "late_rate": round(finding.seg_rate, 4),
            "baseline_rate": round(finding.baseline_rate, 4),
            "lift": round(finding.lift, 3),
            "excess_orders": round(finding.excess_orders, 1),
            "excess_margin_usd": round(finding.excess_margin, 2),
            "confidence": round(finding.confidence, 3),
        }
        try:
            async def call():
                return await self._provider.generate_structured_output(
                    system=mitigation_v1.SYSTEM,
                    user=mitigation_v1.build_user_prompt(facts),
                    schema=_MitigationSchema,
                    max_tokens=500,
                )

            (out, result), retries = await with_retries(
                call, max_retries=self._settings.llm_max_retries, breaker=self._breaker
            )
            metrics.LLM_COST.labels(result.provider, result.model).inc(
                result.usage.estimated_cost_usd
            )
            log_event(logger, "llm_request_completed", correlation_id=correlation_id,
                      agent_name="mitigation", model_provider=result.provider,
                      model_name=result.model, prompt_version=mitigation_v1.PROMPT_VERSION,
                      retry_count=retries)
            finding.narrative = out.narrative
            finding.mitigation = out.mitigation
            finding.expected_effect = out.expected_effect
        except ProviderError as exc:
            # Deterministic fallback narrative: purely descriptive, no invented cause.
            log_event(logger, "fallback_response_used", status="ok",
                      correlation_id=correlation_id, agent_name="mitigation",
                      error_code=type(exc).__name__)
            finding.narrative = (
                f"Observed fact: segment {finding.label} shows a late rate of "
                f"{finding.seg_rate:.1%} versus a baseline of {finding.baseline_rate:.1%} "
                f"across {finding.n} orders ({finding.excess_orders:+.0f} excess late orders)."
            )
            finding.mitigation = (
                "Hypothesis (requires operational validation): review the SLA promise or "
                "service mix for this shipping lane."
            )
            finding.expected_effect = "Not estimated (LLM unavailable)."
        return finding