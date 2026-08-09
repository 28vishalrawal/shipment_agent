"""Agentic layer tests: tool execution, ReAct loop, approval gating."""
from __future__ import annotations

import pytest

from app.agentic.context import RunContext
from app.agentic.react_agent import ReactAgent
from app.analytics.ingest import ingest
from app.approval.store import ApprovalStore
from app.core.config import Settings
from app.providers.factory import build_provider
from app.tools.registry import build_registry
from scripts.make_synthetic_data import generate


def test_tools_are_deterministic_and_registered():
    reg = build_registry()
    assert "run_root_cause_analysis" in reg.names()
    assert "propose_customer_notification" in reg.names()
    # action tools require approval; analysis tools do not
    assert reg.get("propose_ops_escalation").requires_approval
    assert not reg.get("summarize_data").requires_approval


def test_summarize_tool_uses_real_data():
    reg = build_registry()
    df = generate(2000, seed=2)
    ctx = RunContext(run_id="r", correlation_id="c", ingested=ingest(df))
    out = reg.get("summarize_data").invoke({}, ctx)
    assert out["input_rows"] == 2000
    assert 0 <= out["global_late_rate"] <= 1


@pytest.mark.asyncio
async def test_react_agent_follows_scripted_plan_and_queues_approval():
    # Mock provider reads a plan from the system prompt after <<PLAN>>.
    plan = (
        "summarize_data {}\n"
        'score_triage_queue {"n": 3}\n'
        'propose_customer_notification {"order_id": "__PICK__"}'
    )
    df = generate(3000, seed=4)
    ctx = RunContext(run_id="r", correlation_id="c", ingested=ingest(df))

    reg = build_registry()
    settings = Settings(llm_provider="mock")
    provider = build_provider(settings)

    # Pre-score so we know a valid order_id to substitute into the plan.
    from app.analytics.rate_table import build_rate_table
    from app.analytics.triage import score_open_orders
    rt = build_rate_table(ctx.ingested.closed)
    recs = score_open_orders(ctx.ingested.open_orders, rt, 50, 75, 200)
    if not recs:
        pytest.skip("no open orders in sample")
    plan = plan.replace("__PICK__", recs[0].order_id)

    agent = ReactAgent(provider, reg, settings,
                       allowed_tools=["summarize_data", "score_triage_queue",
                                      "propose_customer_notification"],
                       max_steps=8)
    system = "You are a test agent. <<PLAN>>\n" + plan
    trace = await agent.run(system, "do the plan", ctx)

    assert trace.tool_calls_made == 3
    # The side-effecting tool was queued, not executed.
    assert len(ctx.pending_approvals) == 1
    assert ctx.pending_approvals[0]["status"] == "pending_approval"


def test_approval_store_lifecycle():
    store = ApprovalStore()
    item = store.enqueue("run1", "send_notification", {"order_id": "5"})
    assert store.list_pending("run1")
    decided = store.decide(item.id, approve=True, actor="ops")
    assert decided.status.value == "approved"
    store.mark_executed(item.id)
    assert not store.list_pending("run1")
