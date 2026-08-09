"""Lane A notification agent. LLM drafts; guardrails gate; fallback guarantees output."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from app.agents.reliability import CircuitBreaker, with_retries
from app.core import column_mapping as cm
from app.core.config import Settings
from app.domain.models import NotificationRecord, RemedyTier, TriageRecord
from app.guardrails.input_guard import build_allowlisted_payload
from app.guardrails.output_guard import check_notification, fallback_notification
from app.observability.logging_setup import log_event
from app.observability import metrics
from app.prompts import notification_v1
from app.providers.base import LLMProvider, ProviderError

logger = logging.getLogger("agent.notification")

# Only these fields may ever reach the model. Everything else is withheld.
ALLOWLIST = [cm.ORDER_ID, cm.CATEGORY, cm.CUSTOMER_SEGMENT, "product_name", "quantity"]


class _DraftSchema(BaseModel):
    subject: str
    body: str


def _tier(record: TriageRecord) -> RemedyTier:
    if record.value_at_risk > 500 or record.segment == "Corporate":
        return RemedyTier.PARTIAL_SHIP_CONTACT
    if record.value_at_risk >= 100:
        return RemedyTier.EXPEDITE_OFFER
    return RemedyTier.APOLOGY_ONLY


class NotificationAgent:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._breaker = CircuitBreaker()

    async def draft(
        self, record: TriageRecord, source_row: dict, correlation_id: str
    ) -> NotificationRecord:
        tier = _tier(record)
        product = str(source_row.get("product_name", source_row.get(cm.CATEGORY, "your item")))
        quantity = str(source_row.get("quantity", "")).strip()

        payload, in_guard = build_allowlisted_payload(
            {**source_row, "product_name": product, "quantity": quantity}, ALLOWLIST
        )
        payload.update(
            {
                "revised_eta": record.revised_eta.isoformat(),
                "remedy_tier": int(tier),
                "reason_code": record.reason_code.value,
            }
        )
        grounded = {
            "order_id": record.order_id,
            "product_name": product,
            "revised_eta": record.revised_eta.isoformat(),
            "quantity": quantity,
        }

        if not self._settings.enable_llm_notifications:
            return self._fallback(record, product, quantity, tier, "llm_disabled")

        try:
            async def call():
                return await self._provider.generate_structured_output(
                    system=notification_v1.SYSTEM,
                    user=notification_v1.build_user_prompt(payload),
                    schema=_DraftSchema,
                    max_tokens=400,
                )

            (draft, result), retries = await with_retries(
                call, max_retries=self._settings.llm_max_retries, breaker=self._breaker
            )
            metrics.LLM_COST.labels(result.provider, result.model).inc(
                result.usage.estimated_cost_usd
            )
            log_event(
                logger, "llm_request_completed", correlation_id=correlation_id,
                agent_name="notification", model_provider=result.provider,
                model_name=result.model, prompt_version=notification_v1.PROMPT_VERSION,
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                estimated_cost_usd=result.usage.estimated_cost_usd, retry_count=retries,
            )
        except ProviderError as exc:
            log_event(logger, "llm_request_failed", status="error",
                      correlation_id=correlation_id, agent_name="notification",
                      error_code=type(exc).__name__)
            return self._fallback(record, product, quantity, tier, "llm_failed")

        check = check_notification(
            body=draft.body, subject=draft.subject, revised_eta=record.revised_eta,
            grounded_fields=grounded,
        )
        if not check.ok:
            for reason in check.reasons:
                metrics.GUARDRAIL_BLOCKS.labels("output", reason).inc()
            log_event(logger, "customer_notification_guardrail_failed", status="blocked",
                      correlation_id=correlation_id, order_hash=_hash(record.order_id),
                      guardrail_outcome=",".join(check.reasons))
            return self._fallback(record, product, quantity, tier, "guardrail_failed")

        log_event(logger, "customer_notification_generated",
                  correlation_id=correlation_id, order_hash=_hash(record.order_id),
                  guardrail_outcome="pass", remedy_tier=int(tier))
        return NotificationRecord(
            order_id=record.order_id, subject=draft.subject, body=draft.body,
            remedy_tier=tier, grounded_fields=list(grounded.keys()),
            validator_pass=True, used_fallback=False,
        )

    def _fallback(self, record, product, quantity, tier, why) -> NotificationRecord:
        metrics.GUARDRAIL_BLOCKS.labels("output", f"fallback:{why}").inc()
        subject, body = fallback_notification(record.order_id, product, record.revised_eta, quantity)
        return NotificationRecord(
            order_id=record.order_id, subject=subject, body=body, remedy_tier=tier,
            grounded_fields=["order_id", "product_name", "revised_eta"],
            validator_pass=True, used_fallback=True,
        )


def _hash(order_id: str) -> str:
    import hashlib
    return hashlib.sha256(order_id.encode()).hexdigest()[:12]
