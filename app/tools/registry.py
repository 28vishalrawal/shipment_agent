"""Concrete tools. Read/analysis tools are deterministic and free to call.
Action tools (send_notification, file_escalation) are marked requires_approval
so the agent can only PROPOSE them; a human confirms before execution.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agentic.context import RunContext
from app.analytics.rate_table import build_rate_table
from app.analytics.root_cause import GateParams, run_root_cause
from app.analytics.triage import score_open_orders
from app.core import column_mapping as cm
from app.core.config import get_settings
from app.tools.base import Tool, ToolError, ToolRegistry


# ---------------- argument schemas ----------------
class NoArgs(BaseModel):
    pass


class TopNArgs(BaseModel):
    n: int = Field(default=10, ge=1, le=200)


class SegmentArgs(BaseModel):
    dimension: str = Field(description="canonical column, e.g. shipping_mode")
    value: str


class DropModeArgs(BaseModel):
    exclude_shipping_mode: bool = False


class ProposeNotificationArgs(BaseModel):
    order_id: str


class ProposeEscalationArgs(BaseModel):
    finding_label: str
    justification: str = Field(min_length=10)


# ---------------- tool bodies (deterministic) ----------------
def _summarize_data(_: NoArgs, ctx: RunContext) -> dict[str, Any]:
    ing = ctx.ingested
    if ing is None:
        raise ToolError("data not ingested")
    return {
        "input_rows": ing.input_rows,
        "analysis_rows": ing.analysis_rows,
        "open_orders": len(ing.open_orders),
        "global_late_rate": round(ing.global_late_rate, 4),
        "data_quality_flags": ing.data_quality_flags,
        "columns_resolved": list(ing.resolved_columns.keys()),
    }


def _segment_late_rate(a: SegmentArgs, ctx: RunContext) -> dict[str, Any]:
    df = ctx.ingested.closed
    if a.dimension not in df.columns:
        raise ToolError(f"unknown dimension {a.dimension}")
    sub = df[df[a.dimension] == a.value]
    if sub.empty:
        return {"dimension": a.dimension, "value": a.value, "n": 0, "late_rate": None}
    return {
        "dimension": a.dimension,
        "value": a.value,
        "n": int(len(sub)),
        "late_rate": round(float(sub["late"].mean()), 4),
        "baseline_global": round(float(df["late"].mean()), 4),
    }


def _run_root_cause(a: DropModeArgs, ctx: RunContext) -> dict[str, Any]:
    s = get_settings()
    df = ctx.ingested.closed
    avg_margin = float(df[cm.BENEFIT_PER_ORDER].mean()) if cm.BENEFIT_PER_ORDER in df.columns else 0.0
    dims = None
    if a.exclude_shipping_mode:
        from app.analytics.root_cause import DEFAULT_DIMS
        dims = [d for d in DEFAULT_DIMS if d != cm.SHIPPING_MODE]
    out = run_root_cause(
        df,
        GateParams(
            support_floor=s.support_floor, effect_size_min=s.effect_size_min,
            fdr_q=s.fdr_q, confound_margin=s.confound_margin,
            stability_var_max=s.stability_var_max,
        ),
        dims=dims, avg_margin=avg_margin,
    )
    ctx.scratch["root_cause_output"] = out
    return {
        "candidates_enumerated": out.candidates_enumerated,
        "m_tests_conducted": out.m_tests_conducted,
        "findings": [
            {"label": f.label, "n": f.n, "seg_rate": round(f.seg_rate, 4),
             "lift": round(f.lift, 3), "excess_orders": round(f.excess_orders, 1),
             "confidence": round(f.confidence, 3), "evidence_grade": f.evidence_grade}
            for f in out.findings
        ],
        "rejected_count": len(out.rejected),
    }


def _score_triage(a: TopNArgs, ctx: RunContext) -> dict[str, Any]:
    if ctx.rate_table is None:
        ctx.rate_table = build_rate_table(ctx.ingested.closed)
    s = get_settings()
    recs = score_open_orders(
        ctx.ingested.open_orders, ctx.rate_table,
        shrinkage_k=s.shrinkage_k, eta_percentile=s.eta_percentile,
        queue_cap=s.triage_queue_cap,
    )
    ctx.triage = recs
    top = recs[: a.n]
    return {
        "total_flagged": len(recs),
        "top": [
            {"order_id": r.order_id, "p_late": r.p_late, "impact_score": r.impact_score,
             "reason_code": r.reason_code.value, "value_at_risk": r.value_at_risk}
            for r in top
        ],
    }


# ---------------- action tools (require approval) ----------------
def _propose_notification(a: ProposeNotificationArgs, ctx: RunContext) -> dict[str, Any]:
    match = next((r for r in ctx.triage if r.order_id == a.order_id), None)
    if match is None:
        raise ToolError(f"order {a.order_id} not in triage queue")
    action = {
        "type": "send_notification",
        "order_id": a.order_id,
        "p_late": match.p_late,
        "revised_eta": match.revised_eta.isoformat(),
        "status": "pending_approval",
    }
    ctx.pending_approvals.append(action)
    return {"proposed": action, "note": "queued for human approval; not sent"}


def _propose_escalation(a: ProposeEscalationArgs, ctx: RunContext) -> dict[str, Any]:
    action = {
        "type": "file_escalation",
        "finding_label": a.finding_label,
        "justification": a.justification,
        "status": "pending_approval",
    }
    ctx.pending_approvals.append(action)
    return {"proposed": action, "note": "queued for human approval; not filed"}


# ---------------- registry assembly ----------------
def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("summarize_data", "Return row counts, global late rate, and data-quality flags.", NoArgs, _summarize_data))
    reg.register(Tool("segment_late_rate", "Late rate + n for one dimension value vs global baseline.", SegmentArgs, _segment_late_rate))
    reg.register(Tool("run_root_cause_analysis", "Run the deterministic five-gate systemic root-cause pipeline. Set exclude_shipping_mode to search non-mode structure.", DropModeArgs, _run_root_cause))
    reg.register(Tool("score_triage_queue", "Score open orders for late risk and return the top N by impact.", TopNArgs, _score_triage))
    reg.register(Tool("propose_customer_notification", "Propose sending a delay notification for one order. Requires human approval.", ProposeNotificationArgs, _propose_notification, requires_approval=True))
    reg.register(Tool("propose_ops_escalation", "Propose filing an internal escalation for a validated systemic finding. Requires human approval.", ProposeEscalationArgs, _propose_escalation, requires_approval=True))
    return reg
