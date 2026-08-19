"""ReAct-style autonomous agent.

The LLM reasons and chooses tools in a loop until it produces a final answer or
hits the step budget. Deterministic tools do all computation. Any tool marked
requires_approval is NOT executed — it is recorded as a pending action and the
agent is told it was queued, keeping a human in the loop for every side effect.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.agentic.context import RunContext
from app.agents.reliability import CircuitBreaker, with_retries
from app.core.config import Settings
from app.observability.logging_setup import log_event
from app.observability import metrics
from app.providers.base import LLMProvider, ProviderError
from app.tools.base import ToolError, ToolRegistry

logger = logging.getLogger("agentic.react")


@dataclass
class AgentTrace:
    steps: list[dict] = field(default_factory=list)
    final_answer: str = ""
    tool_calls_made: int = 0
    approvals_queued: int = 0


class ReactAgent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        *,
        allowed_tools: list[str] | None = None,
        max_steps: int = 8,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._settings = settings
        self._allowed = allowed_tools or registry.names()
        self._max_steps = max_steps
        self._breaker = CircuitBreaker()

    async def run(self, system_prompt: str, goal: str, ctx: RunContext) -> AgentTrace:
        trace = AgentTrace()
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]
        tool_schemas = self._registry.schemas(self._allowed)

        for step in range(self._max_steps):
            try:
                async def call():
                    return await self._provider.generate_with_tools(
                        messages=messages, tools=tool_schemas, max_tokens=4096
                    )

                result, _ = await with_retries(
                    call, max_retries=self._settings.llm_max_retries, breaker=self._breaker
                )
            except ProviderError as exc:
                log_event(logger, "llm_request_failed", status="error",
                          correlation_id=ctx.correlation_id, agent_name="react",
                          error_code=type(exc).__name__)
                trace.final_answer = "agent_failed: provider unavailable"
                return trace

            metrics.LLM_COST.labels(result.provider, result.model).inc(
                result.usage.estimated_cost_usd
            )

            if not result.tool_calls:
                trace.final_answer = result.text
                log_event(logger, "agentic_final_answer", correlation_id=ctx.correlation_id,
                          steps=step, tool_calls=trace.tool_calls_made)
                return trace

            # Record the assistant turn (with its tool call requests).
            messages.append({
                "role": "assistant",
                "content": result.text or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in result.tool_calls
                ],
            })

            for tc in result.tool_calls:
                trace.tool_calls_made += 1
                obs = await self._execute(tc, ctx, trace)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": tc.name,
                    "content": json.dumps(obs, default=str),
                })

        trace.final_answer = "agent_stopped: step budget exhausted"
        log_event(logger, "agentic_budget_exhausted", correlation_id=ctx.correlation_id,
                  tool_calls=trace.tool_calls_made)
        return trace

    async def _execute(self, tc, ctx: RunContext, trace: AgentTrace) -> dict:
        if tc.name not in self._allowed:
            return {"error": f"tool {tc.name} not permitted for this agent"}
        try:
            tool = self._registry.get(tc.name)
        except ToolError as exc:
            return {"error": str(exc)}

        # Approval gate: side-effecting tools are proposed, never executed here.
        if tool.requires_approval:
            trace.approvals_queued += 1
            log_event(logger, "action_queued_for_approval", correlation_id=ctx.correlation_id,
                      tool=tc.name)
            try:
                observation = tool.invoke(tc.arguments, ctx)  # records pending action
            except ToolError as exc:
                return {"error": str(exc)}
            return observation

        try:
            observation = tool.invoke(tc.arguments, ctx)
        except ToolError as exc:
            return {"error": str(exc)}
        trace.steps.append({"tool": tc.name, "args": tc.arguments})
        log_event(logger, "agentic_tool_executed", correlation_id=ctx.correlation_id,
                  tool=tc.name)
        return observation