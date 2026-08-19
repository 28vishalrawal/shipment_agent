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
        # Once the provider is confirmed down for this run, stop attempting LLM
        # calls for every remaining order — go straight to the personalized
        # deterministic template. Avoids per-order circuit_open noise and latency.
        self._llm_down = False
        # Cache of drafted message SHAPES keyed by template_key(). At scale, many
        # orders share the same (tier, reason, slip) shape, so we draft each shape
        # once with the LLM and fill per-order specifics deterministically. This
        # turns N late orders into (number of distinct shapes) LLM calls.
        self._template_cache: dict[str, tuple[str, str]] = {}

    @staticmethod
    def template_key(record: TriageRecord) -> str:
        """The fields that determine message WORDING (not per-order specifics like
        order_id / product / date, which are slotted in afterwards)."""
        tier = _tier(record)
        return f"tier={int(tier)}|reason={record.reason_code.value}|slip={record.expected_slip_days}"

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
        # Grounding set: drop empty fields (e.g. blank quantity) so they don't
        # count against the achievable grounding total in the output guardrail.
        grounded = {
            k: v for k, v in {
                "order_id": record.order_id,
                "product_name": product,
                "revised_eta": record.revised_eta.isoformat(),
                "quantity": quantity,
            }.items() if v and str(v).strip()
        }
        # Skip the LLM entirely if disabled, or if it already failed this run.
        if not self._settings.enable_llm_notifications or self._llm_down:
            why = "llm_disabled" if not self._settings.enable_llm_notifications else "llm_down_this_run"
            return self._fallback(record, product, quantity, tier, why)
        try:
            async def call():
                return await self._provider.generate_structured_output(
                    system=notification_v1.SYSTEM,
                    user=notification_v1.build_user_prompt(payload),
                    schema=_DraftSchema,
                    max_tokens=2048,
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
            # First confirmed provider failure latches the run into fallback mode
            # so the remaining orders don't each re-attempt a dead provider.
            self._llm_down = True
            log_event(
                logger,
                "llm_request_failed",
                status="error",
                correlation_id=correlation_id,
                agent_name="notification",
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
                status_code=getattr(exc, "status_code", None),
                provider_name=getattr(self._provider, "name", "unknown"),
                model_name=getattr(self._provider, "model", "unknown"),
                prompt_version=notification_v1.PROMPT_VERSION,
                llm_disabled_for_run=True,
            )
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


def fill_template_for_order(
    template_subject: str,
    template_body: str,
    representative: TriageRecord,
    representative_product: str,
    record: TriageRecord,
    product: str,
    quantity: str,
) -> NotificationRecord:
    """Take a drafted message SHAPE (from one representative order) and produce a
    per-order message by substituting this order's specifics. Deterministic — no
    LLM. Guarantees the order's own id / product / ETA appear verbatim so the
    result is personalized, not identical boilerplate."""
    tier = _tier(record)
    subject = template_subject
    body = template_body
    # Replace the representative's specifics with this order's specifics.
    subs = [
        (representative.order_id, record.order_id),
        (representative_product, product),
        (representative.revised_eta.isoformat(), record.revised_eta.isoformat()),
    ]
    for old, new in subs:
        if old and new and old != new:
            subject = subject.replace(old, new)
            body = body.replace(old, new)
    # Safety net: a representative draft with an empty subject/body would
    # otherwise propagate an empty message to the whole pooled group. Never
    # emit a blank message — synthesize a safe, personalized one.
    if not subject or not subject.strip():
        subject = f"Update on your order {record.order_id}"
    if not body or not body.strip():
        body = (f"Hello,\n\nYour order {record.order_id} containing {product} is now "
                f"expected to arrive by {record.revised_eta.isoformat()}. We apologise "
                f"for the delay and appreciate your patience.\n\nKind regards,\nCustomer Care")
    # Safety net: ensure this order's id and revised date are present verbatim.
    if record.order_id not in body:
        body += f"\n\nOrder reference: {record.order_id}."
    if record.revised_eta.isoformat() not in body:
        body += f" Revised delivery date: {record.revised_eta.isoformat()}."
    return NotificationRecord(
        order_id=record.order_id, subject=subject, body=body, remedy_tier=tier,
        grounded_fields=["order_id", "product_name", "revised_eta"],
        validator_pass=True, used_fallback=False,
    )