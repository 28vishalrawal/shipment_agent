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
    async def _lane_a(self, ingested, run_id, correlation_id):
        rt = build_rate_table(ingested.closed)
        triage = score_open_orders(
            ingested.open_orders, rt,
            shrinkage_k=self._settings.shrinkage_k,
            eta_percentile=self._settings.eta_percentile,
            queue_cap=self._settings.triage_queue_cap,
        )
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

        notifications: list[NotificationRecord] = []
        # Bounded concurrency to avoid hammering the provider.
        sem = asyncio.Semaphore(10)

        async def one(rec: TriageRecord):
            async with sem:
                try:
                    return await self._notifier.draft(
                        rec, by_id.get(rec.order_id, {}), correlation_id
                    )
                except Exception as exc:  # per-order isolation
                    log_event(logger, "customer_notification_generated", status="error",
                              correlation_id=correlation_id, error_code=type(exc).__name__)
                    return None

        results = await asyncio.gather(*(one(r) for r in triage))
        notifications = [n for n in results if n is not None]
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

        # Escalation gate.
        best = max((f.confidence for f in out.findings), default=0.0)
        escalate = best >= self._settings.escalation_confidence
        decision = EscalationDecision(
            run_id=run_id,
            candidates_evaluated=out.candidates_enumerated,
            m_tests_conducted=out.m_tests_conducted,
            escalated=escalate,
            finding_id=out.findings[0].pattern_id if (escalate and out.findings) else None,
            confidence=best,
            threshold=self._settings.escalation_confidence,
            suppression_reason=None if escalate else (
                f"best confidence {best:.2f} < {self._settings.escalation_confidence}"
            ),
        )
        if escalate:
            metrics.ESCALATIONS.inc()
            log_event(logger, "systemic_risk_detected", run_id=run_id,
                      correlation_id=correlation_id, finding=decision.finding_id,
                      confidence=round(best, 3))
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
