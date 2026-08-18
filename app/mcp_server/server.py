"""Builds the MCP server exposed to sandboxed agents.

Surface
-------
Two families of tools, both read-only:

  * Bridged analytics — every non-approval tool from `app.tools.registry`,
    run against a named dataset (see `app.mcp_server.bridge`).
  * Run history — what the pipeline already produced, from `RunStore`. This is
    what a conversational client usually wants: it asks about last night's
    batch, which has already been analysed, rather than asking for a fresh
    five-gate pass over 180k rows.

Nothing here writes to the domain. Notifications and escalations are proposed
and approved on the authenticated HTTP path, where a principal and a run_id
stand behind the action.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from app.core.config import Settings
from app.mcp_server.bridge import register_registry_tools
from app.mcp_server.datasets import list_datasets
from app.persistence.run_store import get_run_store
from app.tools.registry import build_registry

logger = logging.getLogger("mcp.server")

INSTRUCTIONS = """\
Shipment delay and exception analytics.

Every number these tools return is computed deterministically in Python — late
rates, lifts, p-values and confidence scores are never estimated by a model.
Report them as given and do not recompute, extrapolate or round them further.

Each finding carries an `evidence_grade`:
  observed_fact       — measured directly in the data
  data_supported_risk — inferred from a validated statistical pattern
  hypothesis          — plausible but not established

Never present a hypothesis as established, and never attribute a delay to a
carrier: the source data has no carrier field, only shipping lane and mode.

Use get_recent_runs and get_run_findings for analysis the pipeline has already
completed. Use the dataset tools only when asked about a batch that has not
been run, and call list_datasets first to learn the valid names.
"""

# Agent trajectories are large, repetitive, and meaningless outside a debugger.
# They are stripped from run payloads so one get_run_findings call cannot
# consume a conversational agent's whole context window.
_TRAJECTORY_KEYS = ("triage_agent", "root_cause_agent")


def _strip_trajectories(result: dict[str, Any]) -> dict[str, Any]:
    trimmed = {k: v for k, v in result.items() if k not in _TRAJECTORY_KEYS}
    for key in _TRAJECTORY_KEYS:
        agent = result.get(key)
        if isinstance(agent, dict):
            # The conclusion is worth keeping; the step-by-step is not.
            trimmed[key] = {
                "final_answer": agent.get("final_answer"),
                "tool_calls": agent.get("tool_calls"),
            }
    return trimmed


def build_mcp_server(settings: Settings) -> MCPServer:
    server = MCPServer(
        name=settings.service_name,
        version=settings.build_version,
        instructions=INSTRUCTIONS,
    )

    published = register_registry_tools(server, build_registry(), settings)

    @server.tool(
        name="list_datasets",
        description="List batch files available for analysis, newest first.",
    )
    async def _list_datasets() -> dict:
        rows = await asyncio.to_thread(list_datasets, settings)
        return {"datasets": rows, "count": len(rows)}

    @server.tool(
        name="get_recent_runs",
        description=(
            "List completed analysis runs, newest first, with at-risk order and "
            "escalation counts. Use this before get_run_findings to find a run_id."
        ),
    )
    async def _recent_runs(
        limit: Annotated[int, Field(default=10, ge=1, le=50)] = 10,
        source: Annotated[
            Literal["any", "upload", "webhook", "file_drop", "scheduler"],
            Field(default="any", description="Filter by what triggered the run."),
        ] = "any",
    ) -> dict:
        store = get_run_store()
        rows = await asyncio.to_thread(
            store.list, limit, None if source == "any" else source
        )
        return {"runs": [r.to_dict() for r in rows], "count": len(rows)}

    @server.tool(
        name="get_run_findings",
        description=(
            "Full results for one run: validated root causes with evidence grades, "
            "mitigations, escalations, and pending approvals. Agent step-by-step "
            "traces are omitted."
        ),
    )
    async def _run_findings(
        run_id: Annotated[str, Field(description="run_id from get_recent_runs")],
    ) -> dict:
        store = get_run_store()
        record = await asyncio.to_thread(store.get, run_id)
        if record is None:
            return {"error": f"run {run_id!r} not found"}
        return {
            **record.summary.to_dict(),
            "result": _strip_trajectories(record.result),
        }

    logger.info(
        "mcp_server_built tools=%s", published + [
            "list_datasets", "get_recent_runs", "get_run_findings"
        ],
    )
    return server


def build_mcp_asgi_app(settings: Settings):
    """The Streamable HTTP ASGI app, wrapped in bearer auth.

    Mounted rather than run standalone so the MCP surface shares the service's
    process, logging and metrics — and so a single TLS terminator in front of
    uvicorn covers both the REST API and MCP.
    """
    from app.mcp_server.auth import BearerAuthMiddleware

    server = build_mcp_server(settings)

    allowed = [h.strip() for h in settings.mcp_allowed_hosts.split(",") if h.strip()]
    # DNS-rebinding protection validates the Host header. Behind a TLS
    # terminator that header is the public hostname NemoClaw was given, which
    # this process would otherwise never recognise.
    transport_security = (
        TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed,
            allowed_origins=[f"https://{h}" for h in allowed],
        )
        if allowed
        else None
    )

    app = server.streamable_http_app(
        streamable_http_path="/",
        transport_security=transport_security,
    )
    return server, BearerAuthMiddleware(app, settings.mcp_bearer_token)