"""Orchestrator: fan-out to Lane A + Lane B in parallel, fan-in, escalation gate.

Lane A degrades gracefully (per-order failures are skipped). Lane B fails atomically
(no half-tested systemic claim is ever emitted).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from app.analytics.ingest import IngestResult
from app.analytics.rate_table import build_rate_table
from app.analytics.root_cause import GateParams, run_root_cause
from app.analytics.triage import score_open_orders
from app.agents.mitigation_agent import MitigationAgent
from app.agents.notification_agent import NotificationAgent
from app.core import column_mapping as cm
from app.core.config import Settings
from app.domain.models import (
    EscalationDecision,
    NotificationRecord,
    RootCauseReport,
    TriageRecord,
)
from app.observability.logging_setup import log_event, new_id
from app.observability import metrics
from app.providers.base import LLMProvider


def _decision_for(run_id: str, out, settings: Settings, finding, rank: int) -> EscalationDecision:
    """Wrap one ranked finding in the self-contained EscalationDecision shape,
    applying the confidence gate. `finding` may be None (nothing validated), in
    which case a suppressed decision is returned.
    """
    conf = finding.confidence if finding else 0.0
    escalate = bool(finding) and conf >= settings.escalation_confidence
    return EscalationDecision(
        run_id=run_id,
        candidates_evaluated=out.candidates_enumerated,
        m_tests_conducted=out.m_tests_conducted,
        escalated=escalate,
        rank=rank,
        finding_id=finding.pattern_id if escalate else None,
        confidence=conf,
        threshold=settings.escalation_confidence,
        suppression_reason=None if escalate else (
            f"best confidence {conf:.2f} < {settings.escalation_confidence}"
        ),
        finding_label=finding.label if escalate else None,
        excess_orders=round(finding.excess_orders, 1) if escalate else None,
        excess_margin_usd=round(finding.excess_margin, 2) if escalate else None,
        narrative=finding.narrative if escalate else None,
        mitigation=finding.mitigation if escalate else None,
        expected_effect=finding.expected_effect if escalate else None,
    )


def build_escalation_decisions(
    run_id: str, out, settings: Settings, limit: int | None = None
) -> list[EscalationDecision]:
    """Build up to `limit` escalation decisions — one per ranked root-cause
    finding that clears the confidence gate.

    Findings arrive already ranked by impact (|excess orders| x confidence), so
    taking the first `limit` yields the strongest systemic causes. Only findings
    that individually clear `escalation_confidence` are returned, which keeps the
    alert-fatigue guard intact: a weak second finding is simply omitted rather
    than padded in to fill the slot.

    Callers must have populated narrative/mitigation/expected_effect on each
    surfaced finding (via MitigationAgent) before calling if they want those
    inlined.
    """
    limit = settings.max_escalations if limit is None else limit
    limit = max(0, limit)
    decisions = [
        _decision_for(run_id, out, settings, f, rank)
        for rank, f in enumerate(out.findings[:limit], start=1)
    ]
    return [d for d in decisions if d.escalated]


def build_escalation_decision(run_id: str, out, settings: Settings) -> EscalationDecision:
    """Single top-finding escalation decision, retained for the deterministic
    /analyze pipeline (RootCauseReport.escalation is a single object) and for
    backward compatibility. Always returns a decision — suppressed when the top
    finding misses the gate or nothing validated.
    """
    top_finding = out.findings[0] if out.findings else None
    return _decision_for(run_id, out, settings, top_finding, 1)

logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._settings = settings
        self._notifier = NotificationAgent(provider, settings)
        self._mitigator = MitigationAgent(provider, settings)

    async def run(
        self, ingested: IngestResult, correlation_id: str | None = None
    ) -> tuple[list[TriageRecord], list[NotificationRecord], RootCauseReport]:
        run_id = new_id()
        correlation_id = correlation_id or new_id()
        log_event(logger, "request_received", run_id=run_id, correlation_id=correlation_id,
                  input_rows=ingested.input_rows, analysis_rows=ingested.analysis_rows,
                  data_quality_flags=",".join(ingested.data_quality_flags))

        lane_a = asyncio.create_task(self._lane_a(ingested, run_id, correlation_id))
        lane_b = asyncio.create_task(self._lane_b(ingested, run_id, correlation_id))
        (triage, notifications), report = await asyncio.gather(lane_a, lane_b)
        return triage, notifications, report

    # ---------------- Lane A ----------------
    async def _lane_a(self, ingested, run_id, correlation_id,
                      queue_cap=None, at_risk_only=False):
        # queue_cap=None -> use the configured cap (default 200). Pass 0 to draft
        # for EVERY open order (no cap). at_risk_only drops LOW_RISK orders so we
        # only notify customers whose shipments are actually flagged at-risk.
        rt = build_rate_table(ingested.closed)
        cap = self._settings.triage_queue_cap if queue_cap is None else queue_cap
        triage = score_open_orders(
            ingested.open_orders, rt,
            shrinkage_k=self._settings.shrinkage_k,
            eta_percentile=self._settings.eta_percentile,
            queue_cap=cap,
        )
        if at_risk_only:
            from app.domain.models import ReasonCode
            triage = [t for t in triage if t.reason_code != ReasonCode.LOW_RISK]
        metrics.EXCEPTIONS_DETECTED.inc(len(triage))
        log_event(logger, "triage_classification_completed", run_id=run_id,
                  correlation_id=correlation_id, flagged=len(triage))

        # Map order_id -> source row for grounding (open cohort only).
        by_id = {}
        if cm.ORDER_ID in ingested.open_orders.columns:
            by_id = {
                str(r[cm.ORDER_ID]): r.to_dict()
                for _, r in ingested.open_orders.iterrows()
            }

        # ---- Template pooling: draft ONE message per distinct template shape,
        # then fill every other order in that shape deterministically. Turns N
        # late orders into (distinct shapes) LLM calls — typically < 50 even for
        # millions of orders. Each customer still gets their own id/product/date.
        from app.agents.notification_agent import fill_template_for_order, _tier as _note_tier

        def _product_of(rec: TriageRecord) -> str:
            row = by_id.get(rec.order_id, {})
            return str(row.get("product_name", row.get(cm.CATEGORY, "your item")))

        def _quantity_of(rec: TriageRecord) -> str:
            row = by_id.get(rec.order_id, {})
            return str(row.get("quantity", "")).strip()

        # Group triage records by template key.
        groups: dict[str, list[TriageRecord]] = {}
        for rec in triage:
            groups.setdefault(self._notifier.template_key(rec), []).append(rec)

        log_event(logger, "notification_pooling", run_id=run_id,
                  correlation_id=correlation_id, orders=len(triage),
                  distinct_templates=len(groups))

        notifications: list[NotificationRecord] = []
        sem = asyncio.Semaphore(10)

        async def draft_group(key: str, recs: list[TriageRecord]):
            representative = recs[0]
            rep_product = _product_of(representative)
            async with sem:
                try:
                    rep_note = await self._notifier.draft(
                        representative, by_id.get(representative.order_id, {}), correlation_id
                    )
                except Exception as exc:
                    log_event(logger, "customer_notification_generated", status="error",
                              correlation_id=correlation_id, error_code=type(exc).__name__)
                    return []
            out = [rep_note]
            # Fill the remaining orders in this group from the representative's
            # message shape — deterministic, no further LLM calls.
            if rep_note.used_fallback:
                # Provider down / guardrail fallback: each order gets its own
                # personalized fallback template directly.
                for rec in recs[1:]:
                    out.append(self._notifier._fallback(
                        rec, _product_of(rec), _quantity_of(rec),
                        _note_tier(rec), "pooled_fallback",
                    ))
            else:
                for rec in recs[1:]:
                    out.append(fill_template_for_order(
                        rep_note.subject, rep_note.body, representative, rep_product,
                        rec, _product_of(rec), _quantity_of(rec),
                    ))
            return out

        group_results = await asyncio.gather(
            *(draft_group(k, v) for k, v in groups.items())
        )
        for g in group_results:
            notifications.extend(g)
        return triage, notifications

    # ---------------- Lane B ----------------
    async def _lane_b(self, ingested, run_id, correlation_id) -> RootCauseReport:
        log_event(logger, "analytics_batch_started", run_id=run_id, correlation_id=correlation_id)

        avg_margin = 0.0
        if cm.BENEFIT_PER_ORDER in ingested.closed.columns:
            avg_margin = float(ingested.closed[cm.BENEFIT_PER_ORDER].mean())

        params = GateParams(
            support_floor=self._settings.support_floor,
            effect_size_min=self._settings.effect_size_min,
            fdr_q=self._settings.fdr_q,
            confound_margin=self._settings.confound_margin,
            stability_var_max=self._settings.stability_var_max,
        )
        with metrics.BATCH_DURATION.time():
            out = run_root_cause(ingested.closed, params, avg_margin=avg_margin)

        log_event(logger, "analytics_segment_calculated", run_id=run_id,
                  correlation_id=correlation_id, candidates=out.candidates_enumerated,
                  m_tests=out.m_tests_conducted, findings=len(out.findings),
                  rejected=len(out.rejected))

        # LLM narratives for validated findings only.
        for f in out.findings:
            await self._mitigator.explain(f, correlation_id)

        # Escalation gate (shared builder — identical shape to the agentic path).
        top_finding = out.findings[0] if out.findings else None
        decision = build_escalation_decision(run_id, out, self._settings)
        escalate = decision.escalated
        if escalate:
            metrics.ESCALATIONS.inc()
            log_event(logger, "systemic_risk_detected", run_id=run_id,
                      correlation_id=correlation_id, finding=decision.finding_id,
                      confidence=round(decision.confidence, 3))
        log_event(logger, "escalation_decision_created", run_id=run_id,
                  correlation_id=correlation_id, escalated=escalate,
                  suppression_reason=decision.suppression_reason)

        return RootCauseReport(
            run_id=run_id,
            generated_at=datetime.now(timezone.utc),
            input_rows=ingested.input_rows,
            analysis_rows=ingested.analysis_rows,
            global_late_rate=ingested.global_late_rate,
            candidates_enumerated=out.candidates_enumerated,
            m_tests_conducted=out.m_tests_conducted,
            findings=out.findings,
            rejected=out.rejected,
            escalation=decision,
        )