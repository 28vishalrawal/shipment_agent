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
from app.agents.orchestrator import (
    Orchestrator as DeterministicOrchestrator,
    build_escalation_decision,
    build_escalation_decisions,
)
from app.analytics.ingest import IngestResult
from app.approval.store import get_approval_store
from app.core.config import Settings
from app.observability.logging_setup import log_event, new_id
from app.prompts import agentic_v1
from app.providers.base import LLMProvider
from app.tools.registry import build_registry

logger = logging.getLogger("agentic.orchestrator")

# How many ranked root causes to surface in the agentic response.
TOP_N_ROOT_CAUSES = 10


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
        det = DeterministicOrchestrator(self._provider, self._settings)
        triage_records, notifications = await det._lane_a(
            ingested, run_id, correlation_id, queue_cap=0, at_risk_only=True
        )
        ctx.triage = triage_records
        ctx.notifications = notifications
        draft_by_id = {n.order_id: n for n in notifications}

        # Drop the triage agent's lightweight notification proposals: they carry
        # only order_id/p_late (no drafted subject/body/tier), which is exactly
        # why those orders showed up empty on the approval page. The deterministic
        # coverage below drafts EVERY at-risk order in full and owns the queue.
        ctx.pending_approvals = [
            a for a in ctx.pending_approvals if a.get("type") != "send_notification"
        ]
        for rec in triage_records:
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
        # Authoritative root-cause report: ALWAYS computed with the full
        # dimension set (identical to /analyze's _lane_b), independent of any
        # exclude_shipping_mode experiments the root-cause agent ran via its
        # tool. Previously this used the agent's scratch output, so a
        # mode-excluded agent run hid 'shipping_mode=First Class' and produced
        # 0 root causes / no escalation while /analyze escalated it.
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

        top = rc_out.findings[0] if rc_out.findings else None
        mitigator = MitigationAgent(self._provider, self._settings)

        async def _explain(f):
            if f.narrative is None:
                await mitigator.explain(f, correlation_id)

        # Validation status per candidate pattern, for labelling.
        status_by_pid = {f.pattern_id: "validated" for f in rc_out.findings}
        for r in rc_out.rejected:
            status_by_pid.setdefault(r.pattern_id, r.failed_gate)

        # Ranked root causes for the UI: validated findings first (with
        # mitigation), then fill up to TOP_N with the strongest candidate
        # segments so the panel is never empty when signals exist but none
        # cleared full validation. Candidates are clearly graded 'hypothesis'.
        root_causes: list[dict] = []
        for rank, f in enumerate(rc_out.findings[:TOP_N_ROOT_CAUSES], start=1):
            await _explain(f)
            row = {"rank": rank, "dimensions": f.dims, "status": "validated"}
            row.update(_escalation_fields(f))
            root_causes.append(row)

        if len(root_causes) < TOP_N_ROOT_CAUSES:
            seen = {r.get("finding_label") for r in root_causes}
            for c in rc_out.top_candidates:
                if len(root_causes) >= TOP_N_ROOT_CAUSES:
                    break
                if c["pattern_id"] in seen:
                    continue
                gate = status_by_pid.get(c["pattern_id"], "candidate")
                root_causes.append({
                    "rank": len(root_causes) + 1,
                    "finding_label": c["pattern_id"],
                    "dimensions": c["dims"],
                    "status": gate,
                    "n": c["n"],
                    "late_rate": round(c["rate"], 4),
                    "baseline_rate": round(c["base"], 4),
                    "lift": round(c["lift"], 3),
                    "excess_orders": round(c["excess"], 1),
                    "excess_margin_usd": None,
                    "confidence": None,
                    "evidence_grade": "hypothesis",
                    "narrative": None,
                    "mitigation": (f"Candidate signal — did not clear full validation "
                                   f"({gate}). Investigate before acting."),
                    "expected_effect": None,
                })

        # --- Escalation: reuse the SAME EscalationDecision the /analyze pipeline
        # emits (build_escalation_decision), so the UI shows identical, rich,
        # self-contained detail (confidence, threshold, excess_orders,
        # excess_margin_usd, narrative, mitigation, expected_effect). The
        # deterministic confidence gate on the top validated finding is
        # authoritative; the agent's ad-hoc free-text escalations are dropped.
        # Up to settings.max_escalations findings may escalate, so every one of
        # them needs its narrative/mitigation inlined before the decisions are
        # built (not just the single top finding).
        for f in rc_out.findings[:max(0, self._settings.max_escalations)]:
            await _explain(f)
        decisions = build_escalation_decisions(run_id, rc_out, self._settings)
        escalations = [d.model_dump() for d in decisions]

        # `escalation` (singular) stays in the payload for backward compatibility:
        # the top decision when one escalated, otherwise the suppressed decision
        # carrying the reason the gate held.
        if top is not None:
            await _explain(top)
        top_decision = build_escalation_decision(run_id, rc_out, self._settings)
        escalation = escalations[0] if escalations else top_decision.model_dump()

        ctx.pending_approvals = [
            a for a in ctx.pending_approvals if a.get("type") != "file_escalation"
        ]
        for e in escalations:
            ctx.pending_approvals.append(
                {"type": "file_escalation", "status": "pending_approval", **e}
            )
        log_event(logger, "escalation_decision_created", run_id=run_id,
                  correlation_id=correlation_id,
                  escalated=bool(escalations), escalation_count=len(escalations),
                  finding_ids=[e.get("finding_id") for e in escalations],
                  suppression_reason=top_decision.suppression_reason)

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
            "escalation": escalation,
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