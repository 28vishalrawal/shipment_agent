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
from app.analytics.ingest import IngestResult
from app.approval.store import get_approval_store
from app.core.config import Settings
from app.observability.logging_setup import log_event, new_id
from app.prompts import agentic_v1
from app.providers.base import LLMProvider
from app.tools.registry import build_registry

logger = logging.getLogger("agentic.orchestrator")


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

        # Move proposed actions into the approval store (human-in-the-loop gate).
        store = get_approval_store()
        queued = []
        for action in ctx.pending_approvals:
            item = store.enqueue(run_id, action["type"], action)
            queued.append({"approval_id": item.id, "type": item.action_type,
                           "status": item.status.value})

        log_event(logger, "agentic_run_completed", run_id=run_id,
                  correlation_id=correlation_id,
                  triage_tool_calls=triage_trace.tool_calls_made,
                  rc_tool_calls=rc_trace.tool_calls_made,
                  actions_queued=len(queued))

        return {
            "run_id": run_id,
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
