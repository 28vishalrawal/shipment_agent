"""Adapter that publishes `app.tools.registry` tools onto an MCP server.

The exposure rule
-----------------
A tool is published if and only if `requires_approval` is False. This is a
structural filter, not a maintained allowlist: any action tool added to the
registry later is excluded automatically, and making one reachable over MCP
requires deliberately clearing the flag that also gates it everywhere else.

That matters because the approval gate lives outside this process boundary. A
sandboxed agent proposing `send_notification` over MCP would enqueue work that
the agent itself cannot execute, so the only thing publishing it buys is a way
for a chat client to fill the human review queue. Proposals stay on the HTTP
path, where the proposer is an authenticated principal with a run behind them.

Schema fidelity
---------------
Each tool's Pydantic `args_schema` is projected onto the wrapper's signature so
the MCP tool schema carries the same field names, types, defaults, descriptions
and constraints (`ge`, `le`, `min_length`) the HTTP path enforces. Flattening
the fields rather than nesting the model keeps the schema shallow, which small
tool-calling models handle far more reliably.
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import time
from typing import Annotated, Any

from pydantic import BaseModel

from app.core.config import Settings
from app.mcp_server.datasets import DatasetError, get_cache
from app.tools.base import Tool, ToolError, ToolRegistry

logger = logging.getLogger("mcp.bridge")

# Monotonic per-process counter so log lines can be read back in the exact
# order the agent called tools, even when calls overlap.
_call_seq = itertools.count(1)

EXPOSURE_RULE = "requires_approval is False"

# Appended to every bridged description so the calling model knows the argument
# exists and what it selects. The registry's own descriptions predate MCP and
# say nothing about datasets.
_DATASET_HINT = (
    " Requires `dataset`: a batch file name from list_datasets, or 'latest' "
    "for the most recent."
)


def exposed_tools(registry: ToolRegistry) -> list[Tool]:
    """Registry tools eligible for MCP publication, in registration order."""
    return [
        registry.get(name)
        for name in registry.names()
        if not registry.get(name).requires_approval
    ]


def _signature_from(model: type[BaseModel]) -> inspect.Signature:
    """Flatten a Pydantic model's fields into a keyword-only signature.

    `Annotated[type, FieldInfo]` is what carries the constraints and description
    through to the generated JSON schema; passing the bare annotation would emit
    a schema that accepts values the tool then rejects at validation time.
    """
    params = [
        inspect.Parameter("dataset", inspect.Parameter.KEYWORD_ONLY, annotation=str)
    ]
    for name, field in model.model_fields.items():
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[field.annotation, field],
                default=inspect.Parameter.empty if field.is_required() else field.default,
            )
        )
    return inspect.Signature(params, return_annotation=dict)


def _make_handler(tool: Tool, settings: Settings):
    """Build the async MCP handler for one registry tool."""

    async def handler(**kwargs: Any) -> dict:
        dataset = kwargs.pop("dataset")
        cache = get_cache(settings)

        # One id per call so the start line and finish line can be matched, and
        # so overlapping calls stay distinguishable when read in order.
        seq = next(_call_seq)
        logger.info(
            "mcp_tool_call seq=%d tool=%s dataset=%s args=%s",
            seq, tool.name, dataset, dict(kwargs),
        )
        started = time.perf_counter()

        def run() -> Any:
            ctx = cache.get(settings, dataset)
            return tool.invoke(kwargs, ctx)

        try:
            # ingest/pandas/scipy are synchronous and CPU-bound; off the loop so
            # one heavy root-cause call cannot stall concurrent MCP sessions.
            result = await asyncio.to_thread(run)
        except (DatasetError, ToolError, ValueError) as exc:
            # ValueError covers ingest() rejecting a file that parsed as a frame
            # but is not an orders export (missing required columns). Like a bad
            # dataset name, that is a condition the caller can correct by picking
            # a different file — so it is reported, not raised.
            # Surfaced as a normal result rather than an exception: the caller is
            # a model that can correct itself if told the name was wrong, but
            # only sees a generic failure if this raises.
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "mcp_tool_rejected seq=%d tool=%s reason=%s elapsed_ms=%.1f",
                seq, tool.name, exc, elapsed_ms,
            )
            return {"error": str(exc), "tool": tool.name}
        except Exception:
            # An unexpected fault (not caller-correctable) would otherwise be
            # swallowed into a generic MCP ToolError with no trace. Log it with
            # the sequence id and full traceback, then re-raise unchanged.
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "mcp_tool_error seq=%d tool=%s elapsed_ms=%.1f",
                seq, tool.name, elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "mcp_tool_ok seq=%d tool=%s dataset=%s elapsed_ms=%.1f",
            seq, tool.name, dataset, elapsed_ms,
        )
        return {"dataset": dataset, "result": result}

    handler.__name__ = tool.name
    handler.__doc__ = tool.description + _DATASET_HINT
    handler.__signature__ = _signature_from(tool.args_schema)
    handler.__annotations__ = {
        p.name: p.annotation for p in handler.__signature__.parameters.values()
    } | {"return": dict}
    return handler


def register_registry_tools(server, registry: ToolRegistry, settings: Settings) -> list[str]:
    """Publish every eligible registry tool onto `server`. Returns their names."""
    published: list[str] = []
    for tool in exposed_tools(registry):
        server.add_tool(
            _make_handler(tool, settings),
            name=tool.name,
            description=tool.description + _DATASET_HINT,
        )
        published.append(tool.name)

    withheld = [n for n in registry.names() if n not in published]
    logger.info(
        "mcp_tools_published published=%s withheld=%s", published, withheld
    )
    return published