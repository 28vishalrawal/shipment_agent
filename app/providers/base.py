"""Provider-agnostic LLM interface.

Every provider returns an LLMResult carrying usage + model identity so the
observability layer can log tokens, cost, provider and model uniformly.
Business code depends ONLY on this Protocol, never on a vendor SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Optional[Any] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMToolResult:
    """Result of a tool-calling turn: either tool calls to execute, or a final
    text answer when the model stops calling tools."""
    tool_calls: list[ToolCall]
    text: str
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def generate_text(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> LLMResult:
        ...

    async def generate_structured_output(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        max_tokens: int = 512,
    ) -> tuple[T, LLMResult]:
        ...

    async def generate_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1024,
    ) -> LLMToolResult:
        """One turn of a tool-calling conversation. `messages` is the running
        transcript (system/user/assistant/tool roles). Returns tool calls to run,
        or a final text answer with no tool calls."""
        ...

    async def health_check(self) -> bool:
        ...


class ProviderError(RuntimeError):
    """Raised for any provider-side failure (network, auth, malformed output)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
