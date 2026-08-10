"""Concrete providers + factory.

OpenAIProvider is fully implemented. Anthropic/Gemini/Azure are structured stubs
that already conform to the Protocol so swapping is a one-line config change.
MockProvider makes the whole system testable with zero network + zero keys.
"""
from __future__ import annotations

import json
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResult,
    LLMUsage,
    ProviderError,
)

T = TypeVar("T", bound=BaseModel)


def _extract_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _provider_error(exc: Exception) -> ProviderError:
    return ProviderError(str(exc), status_code=_extract_status_code(exc))

# Rough per-1K-token prices (USD) for cost estimation only. Update as needed.
_PRICES = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.0025, 0.010),
}


def _estimate_cost(model: str, pin: int, pout: int) -> float:
    cin, cout = _PRICES.get(model, (0.0, 0.0))
    return round(pin / 1000 * cin + pout / 1000 * cout, 6)


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY missing")
        # Imported lazily so the package is optional for mock-only runs.
        from openai import AsyncOpenAI

        self.name = "openai"
        self.model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_s
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key, timeout=settings.llm_timeout_s
        )

    async def generate_text(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> LLMResult:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # normalise every SDK error
            # Keep the error message available to callers/logs.
            raise _provider_error(exc) from exc
        u = resp.usage
        usage = LLMUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
            estimated_cost_usd=_estimate_cost(
                self.model,
                getattr(u, "prompt_tokens", 0),
                getattr(u, "completion_tokens", 0),
            ),
        )
        return LLMResult(
            text=resp.choices[0].message.content or "",
            model=self.model,
            provider=self.name,
            usage=usage,
            raw=resp,
        )

    async def generate_structured_output(
        self, *, system: str, user: str, schema: Type[T], max_tokens: int = 512
    ) -> tuple[T, LLMResult]:
        # Force JSON; instruct the model that the schema is authoritative.
        sys = (
            system
            + "\n\nReturn ONLY a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise _provider_error(exc) from exc
        content = resp.choices[0].message.content or "{}"
        u = resp.usage
        usage = LLMUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
            estimated_cost_usd=_estimate_cost(
                self.model,
                getattr(u, "prompt_tokens", 0),
                getattr(u, "completion_tokens", 0),
            ),
        )
        result = LLMResult(
            text=content, model=self.model, provider=self.name, usage=usage, raw=resp
        )
        try:
            parsed = schema.model_validate_json(content)
        except ValidationError as exc:
            raise ProviderError(f"schema validation failed: {exc}") from exc
        return parsed, result

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def generate_with_tools(self, *, messages, tools, max_tokens=1024):
        from app.providers.base import LLMToolResult, ToolCall
        import json as _json
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                max_tokens=max_tokens,
                tools=tools,
                messages=messages,
            )
        except Exception as exc:
            raise _provider_error(exc) from exc
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = _json.loads(tc.function.arguments or "{}")
            except _json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        u = resp.usage
        usage = LLMUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
            estimated_cost_usd=_estimate_cost(
                self.model, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)
            ),
        )
        return LLMToolResult(
            tool_calls=calls, text=msg.content or "", model=self.model,
            provider=self.name, usage=usage,
        )


class AnthropicProvider:
    """Stub conforming to the Protocol. Fill in with the Anthropic SDK.

    The messages API and tool/JSON-mode differ from OpenAI, but the surface this
    class must present (generate_text / generate_structured_output / health_check)
    is identical, so no business code changes when this replaces OpenAIProvider.
    """

    def __init__(self, settings: Settings) -> None:
        self.name = "anthropic"
        self.model = settings.llm_model
        self._settings = settings

    async def generate_text(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:  # noqa: D102
        raise ProviderError("AnthropicProvider not yet implemented")

    async def generate_structured_output(self, *, system: str, user: str, schema, max_tokens: int = 512):  # noqa: D102
        raise ProviderError("AnthropicProvider not yet implemented")

    async def health_check(self) -> bool:
        return False


class GeminiProvider:
    """Stub conforming to the Protocol. Fill in with the google-genai SDK."""

    def __init__(self, settings: Settings) -> None:
        self.name = "gemini"
        self.model = settings.llm_model
        self._settings = settings

    async def generate_text(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:  # noqa: D102
        raise ProviderError("GeminiProvider not yet implemented")

    async def generate_structured_output(self, *, system: str, user: str, schema, max_tokens: int = 512):  # noqa: D102
        raise ProviderError("GeminiProvider not yet implemented")

    async def health_check(self) -> bool:
        return False


class MockProvider:
    """Deterministic provider for tests. Echoes a schema-valid object.

    It reads a JSON block the caller embeds after the marker '<<JSON>>' in the
    user prompt, enabling tests to drive exact outputs without a network call.
    """

    def __init__(self, settings: Settings) -> None:
        self.name = "mock"
        self.model = "mock-model"

    async def generate_text(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
        return LLMResult(text="MOCK_TEXT", model=self.model, provider=self.name, usage=LLMUsage())

    async def generate_structured_output(self, *, system: str, user: str, schema: Type[T], max_tokens: int = 512):
        if "<<JSON>>" in user:
            payload = user.split("<<JSON>>", 1)[1].strip()
            obj = schema.model_validate_json(payload)
        else:
            # Build a schema-valid placeholder so downstream code never sees a
            # half-constructed object (defaults are applied via model_validate).
            filler = _placeholder_for(schema)
            obj = schema.model_validate(filler)
            payload = obj.model_dump_json()
        return obj, LLMResult(text=payload, model=self.model, provider=self.name, usage=LLMUsage())

    async def health_check(self) -> bool:
        return True

    async def generate_with_tools(self, *, messages, tools, max_tokens=1024):
        """Scripted tool-caller for tests. Reads a plan the test injects into the
        system message after '<<PLAN>>' as newline-separated 'tool_name {json}'
        steps, emitting one per turn, then a final answer. This lets tests drive
        an exact ReAct trajectory with zero network."""
        from app.providers.base import LLMToolResult, ToolCall
        import json as _json

        plan_src = ""
        for m in messages:
            if m.get("role") == "system" and "<<PLAN>>" in (m.get("content") or ""):
                plan_src = m["content"].split("<<PLAN>>", 1)[1].strip()
        steps = [s for s in plan_src.splitlines() if s.strip()] if plan_src else []
        # count how many tool results already in transcript = current step index
        done = sum(1 for m in messages if m.get("role") == "tool")
        if done < len(steps):
            line = steps[done].strip()
            name, _, arg = line.partition(" ")
            try:
                args = _json.loads(arg) if arg.strip() else {}
            except _json.JSONDecodeError:
                args = {}
            return LLMToolResult(
                tool_calls=[ToolCall(id=f"call_{done}", name=name, arguments=args)],
                text="", model=self.model, provider=self.name, usage=LLMUsage(),
            )
        return LLMToolResult(
            tool_calls=[], text="MOCK_FINAL_ANSWER", model=self.model,
            provider=self.name, usage=LLMUsage(),
        )


def _placeholder_for(schema: Type[T]) -> dict:
    """Produce a minimal schema-valid dict for the mock provider."""
    out: dict = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        if ann is str or ann is None:
            out[name] = f"mock_{name}"
        elif ann is int:
            out[name] = 0
        elif ann is float:
            out[name] = 0.0
        elif ann is bool:
            out[name] = False
        else:
            out[name] = f"mock_{name}"
    return out


def build_provider(settings: Settings) -> LLMProvider:
    match settings.llm_provider:
        case "openai":
            return OpenAIProvider(settings)
        case "anthropic":
            return AnthropicProvider(settings)
        case "gemini":
            return GeminiProvider(settings)
        case "mock":
            return MockProvider(settings)
        case _:
            raise ProviderError(f"unknown provider {settings.llm_provider}")
