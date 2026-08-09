"""Tool abstraction for LLM tool-calling agents.

Design rule that survives the shift to autonomous agents: the LLM decides WHICH
tool to call and with what arguments, but every tool body is deterministic Python
(pandas/scipy) from the existing analytics layer. The model never computes a
metric itself; it orchestrates calls to functions that do. This is what keeps a
ReAct loop auditable and prevents fabricated logistics facts.

Every tool exposes an OpenAI-compatible JSON schema (also usable by Anthropic /
Gemini with light adaptation) and a callable that validates its own arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ValidationError


class ToolError(RuntimeError):
    """Raised when a tool's arguments are invalid or execution fails."""


class ToolContext(Protocol):
    """Shared, per-run state a tool may read/write. Passed to every tool call so
    tools never touch globals. Concrete impl lives in agentic.context."""

    run_id: str
    correlation_id: str


@dataclass
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable[[BaseModel, ToolContext], Any]
    # Whether invoking this tool constitutes a real-world side effect that must
    # pass through the human approval gate (send email, file escalation).
    requires_approval: bool = False

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def invoke(self, raw_args: dict, ctx: ToolContext) -> Any:
        try:
            args = self.args_schema.model_validate(raw_args or {})
        except ValidationError as exc:
            raise ToolError(f"invalid args for {self.name}: {exc}") from exc
        return self.func(args, ctx)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool {name}")
        return self._tools[name]

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        names = names or list(self._tools)
        return [self._tools[n].openai_schema() for n in names]

    def names(self) -> list[str]:
        return list(self._tools)
