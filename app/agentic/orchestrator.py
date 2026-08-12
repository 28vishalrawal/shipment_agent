"""Agentic orchestrator: runs two AUTONOMOUS tool-calling agents in parallel.

Contrast with agents/orchestrator.py (fixed pipeline). Here each agent decides
its own tool sequence. Both agents' proposed side effects are collected into the
approval store; nothing executes without human sign-off.
"""
from __future__ import annotations

import asyncio
import logging

from app.agentic.context import RunContext
from app.agentic.react_agent import ReactAgent
from app.agents.mitigation_agent import MitigationAgent
from app.agents.orchestrator import Orchestrator as DeterministicOrchestrator
from app.analytics.ingest import IngestResult
from app.approval.store import get_approval_store
from app.core.config import Settings
from app.observability.logging_setup import log_event, new_id
from app.prompts import agentic_v1
from app.providers.base import LLMProvider
from app.tools.registry import build_registry

logger = logging.getLogger("agentic.orchestrator")

# How many ranked root causes to surface in the agentic response.
TOP_N_ROOT_CAUSES = 5


def _escalation_fields(f) -> dict:
    """The structured, self-contained escalation payload for a validated finding:
    the observed metrics plus the MitigationAgent's narrative / mitigation /
    expected_effect (same shape the deterministic pipeline emits)."""
    return {
        "finding_label": f.label,
        "n": f.n,
        "late_rate": round(f.seg_rate, 4),
        "baseline_rate": round(f.baseline_rate, 4),
        "lift": round(f.lift, 3),
        "excess_orders": round(f.excess_orders, 1),
        "excess_margin_usd": round(f.excess_margin, 2),
        "confidence": round(f.confidence, 3),
        "evidence_grade": f.evidence_grade,
        "narrative": f.narrative,
        "mitigation": f.mitigation,
        "expected_effect": f.expected_effect,
    }


class AgenticOrchestrator:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._registry = build_registry()

    async def run(self, ingested: IngestResult, correlation_id: str | None = None) -> dict:
        run_id = new_id()
        correlation_id = correlation_id or new_id()
        ctx = RunContext(run_id=run_id, correlation_id=correlation_id, ingested=ingested)

        log_event(logger, "agentic_run_started", run_id=run_id,
                  correlation_id=correlation_id, prompt_version=agentic_v1.PROMPT_VERSION)

        triage_agent = ReactAgent(
            self._provider, self._registry, self._settings,
            allowed_tools=["summarize_data", "segment_late_rate",
                           "score_triage_queue", "propose_customer_notification"],
            max_steps=8,
        )
        root_cause_agent = ReactAgent(
            self._provider, self._registry, self._settings,
            allowed_tools=["summarize_data", "segment_late_rate",
                           "run_root_cause_analysis", "propose_ops_escalation"],
            max_steps=8,
        )

        triage_task = asyncio.create_task(
            triage_agent.run(agentic_v1.TRIAGE_AGENT_SYSTEM,
                             "Triage this batch and propose notifications for the "
                             "highest-impact at-risk shipments.", ctx)
        )
        rc_task = asyncio.create_task(
            root_cause_agent.run(agentic_v1.ROOT_CAUSE_AGENT_SYSTEM,
                                 "Determine whether a systemic delay root cause "
                                 "exists and propose an escalation if warranted.", ctx)
        )
        triage_trace, rc_trace = await asyncio.gather(triage_task, rc_task)

        # --- Guaranteed notification coverage (deterministic) ----------------
        # The autonomous triage agent reasons about the batch, but per-order
        # notification COVERAGE must never depend on how many propose_* calls it
        # happened to make inside its step budget (that is why only a handful of
        # approvals appeared before). We therefore draft a notification for
        # EVERY at-risk order deterministically (template-pooled, so this is a
        # few LLM calls, not one per order) and queue one approval per order,
        # carrying the draft. Orders the agent already proposed are de-duplicated.
        already = {
            a.get("order_id") for a in ctx.pending_approvals
            if a.get("type") == "send_notification"
        }
        det = DeterministicOrchestrator(self._provider, self._settings)
        triage_records, notifications = await det._lane_a(
            ingested, run_id, correlation_id, queue_cap=0, at_risk_only=True
        )
        ctx.triage = triage_records
        ctx.notifications = notifications
        draft_by_id = {n.order_id: n for n in notifications}
        for rec in triage_records:
            if rec.order_id in already:
                continue
            n = draft_by_id.get(rec.order_id)
            ctx.pending_approvals.append({
                "type": "send_notification",
                "order_id": rec.order_id,
                "p_late": rec.p_late,
                "revised_eta": rec.revised_eta.isoformat(),
                "reason_code": rec.reason_code.value,
                "impact_score": rec.impact_score,
                "remedy_tier": int(n.remedy_tier) if n else None,
                "subject": n.subject if n else None,
                "body": n.body if n else None,
                "used_fallback": n.used_fallback if n else None,
                "status": "pending_approval",
            })

        log_event(logger, "notification_coverage_ensured", run_id=run_id,
                  correlation_id=correlation_id,
                  at_risk_orders=len(triage_records),
                  notifications_drafted=len(notifications),
                  total_pending=len(ctx.pending_approvals))

        # --- Escalations: attach structured mitigation + apply confidence gate
        # The root-cause agent proposes an escalation with only a free-text
        # justification. Enrich every proposed escalation with the SAME
        # structured explanation the deterministic pipeline produces
        # (narrative / mitigation / expected_effect + supporting metrics) via the
        # versioned MitigationAgent, so an escalation is self-contained and
        # actionable for ops leadership. Also apply the confidence gate: if the
        # agent proposed nothing but a validated finding clears the threshold,
        # escalate it anyway (mirrors agents/orchestrator.py).
        rc_out = ctx.scratch.get("root_cause_output")
        if rc_out is None:
            # The agent may have skipped leaving its analysis in scratch; compute
            # the validated findings once so escalations can still be grounded.
            from app.analytics.root_cause import GateParams, run_root_cause
            from app.core import column_mapping as cm
            avg_margin = (
                float(ingested.closed[cm.BENEFIT_PER_ORDER].mean())
                if cm.BENEFIT_PER_ORDER in ingested.closed.columns else 0.0
            )
            rc_out = run_root_cause(
                ingested.closed,
                GateParams(
                    support_floor=self._settings.support_floor,
                    effect_size_min=self._settings.effect_size_min,
                    fdr_q=self._settings.fdr_q,
                    confound_margin=self._settings.confound_margin,
                    stability_var_max=self._settings.stability_var_max,
                ),
                avg_margin=avg_margin,
            )
            ctx.scratch["root_cause_output"] = rc_out

        top = rc_out.findings[0] if rc_out.findings else None
        root_causes: list[dict] = []
        if top is not None:
            mitigator = MitigationAgent(self._provider, self._settings)

            def _match_finding(label: str):
                # Prefer a finding whose dimension values all appear in the
                # agent's label; fall back to the top-ranked finding.
                for f in rc_out.findings:
                    if f.dims and all(str(v) in (label or "") for v in f.dims.values()):
                        return f
                return top

            async def _explain(f):
                if f.narrative is None:
                    await mitigator.explain(f, correlation_id)

            escalation_actions = [
                a for a in ctx.pending_approvals if a.get("type") == "file_escalation"
            ]
            for action in escalation_actions:
                f = _match_finding(action.get("finding_label", ""))
                await _explain(f)
                action.update(_escalation_fields(f))

            # Confidence gate for the autonomous path.
            if not escalation_actions and top.confidence >= self._settings.escalation_confidence:
                await _explain(top)
                gated = {
                    "type": "file_escalation",
                    "justification": "auto-escalated: confidence >= threshold",
                    "status": "pending_approval",
                }
                gated.update(_escalation_fields(top))
                ctx.pending_approvals.append(gated)

            # Ranked top-N root causes (with recommended mitigation per cause) so
            # leadership can see the full picture, not only what was escalated.
            for rank, f in enumerate(rc_out.findings[:TOP_N_ROOT_CAUSES], start=1):
                await _explain(f)
                row = {"rank": rank, "dimensions": f.dims}
                row.update(_escalation_fields(f))
                root_causes.append(row)

        escalations = [
            {
                "status": a.get("status"),
                **{k: a.get(k) for k in (
                    "finding_label", "late_rate", "baseline_rate", "lift",
                    "excess_orders", "excess_margin_usd", "confidence",
                    "evidence_grade", "narrative", "mitigation", "expected_effect",
                )},
            }
            for a in ctx.pending_approvals if a.get("type") == "file_escalation"
        ]
        log_event(logger, "escalations_enriched", run_id=run_id,
                  correlation_id=correlation_id, escalations=len(escalations))

        # Move proposed actions into the approval store (human-in-the-loop gate).
        store = get_approval_store()
        queued = []
        for action in ctx.pending_approvals:
            item = store.enqueue(run_id, action["type"], action)
            queued.append({"approval_id": item.id, "type": item.action_type,
                           "order_id": action.get("order_id"),
                           "status": item.status.value})

        log_event(logger, "agentic_run_completed", run_id=run_id,
                  correlation_id=correlation_id,
                  triage_tool_calls=triage_trace.tool_calls_made,
                  rc_tool_calls=rc_trace.tool_calls_made,
                  actions_queued=len(queued))

        return {
            "run_id": run_id,
            "at_risk_orders": len(ctx.triage),
            "notifications_drafted": len(ctx.notifications),
            "approvals_created": len(queued),
            "root_causes": root_causes,
            "escalations": escalations,
            "triage_agent": {
                "final_answer": triage_trace.final_answer,
                "tool_calls": triage_trace.tool_calls_made,
                "trajectory": triage_trace.steps,
            },
            "root_cause_agent": {
                "final_answer": rc_trace.final_answer,
                "tool_calls": rc_trace.tool_calls_made,
                "trajectory": rc_trace.steps,
            },
            "pending_approvals": queued,
        }