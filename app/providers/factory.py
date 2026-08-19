"""Concrete providers + factory.

OpenAIProvider is fully implemented. Anthropic/Gemini/Azure are structured stubs
that already conform to the Protocol so swapping is a one-line config change.
MockProvider makes the whole system testable with zero network + zero keys.
"""
from __future__ import annotations

import json
import re
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
    # Self-hosted models have no per-token price and correctly estimate to 0.0;
    # track tokens/latency rather than cost when running locally.
    cin, cout = _PRICES.get(model, (0.0, 0.0))
    return round(pin / 1000 * cin + pout / 1000 * cout, 6)


_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r".*</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)


def _message_text(msg: Any, strip_reasoning: bool = True) -> str:
    """Content of a chat message with any reasoning trace removed.

    Reasoning models (Nemotron 3, and others with a thinking budget) emit a
    chain-of-thought. When the server runs with a reasoning parser it arrives in
    a separate `reasoning_content` field and `content` is already clean; without
    one it is inlined in `content` wrapped in <think> tags, which breaks every
    downstream json.loads. Strip both shapes so structured output survives
    either server configuration.
    """
    # If the server runs a reasoning parser, the trace is split into
    # reasoning_content and `content` is already clean — return it as-is.
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    text = getattr(msg, "content", None) or ""
    if not strip_reasoning or not text:
        return text.strip()
    if reasoning:
        return text.strip()

    # Inlined trace. Handle all three shapes the parser-less path emits:
    #   1. balanced   <think> ... </think> { json }
    #   2. close-only  ... </think> { json }   (opener consumed as a token)
    #   3. no tags     truncated mid-trace, or bare-prose preamble.
    text = _THINK_BLOCK.sub("", text)
    m = _THINK_CLOSE.search(text)
    if m:
        # Everything up to and including the last close tag is reasoning.
        return text[m.end():].strip()
    # No close tag: if a JSON object is present, _json_payload extracts it; if
    # not, the trace was truncated before any answer — return as-is so the
    # caller's parse fails and the deterministic fallback fires.
    return text.strip()


def _json_payload(text: str) -> str:
    """First balanced JSON object in a string.

    Smaller models often wrap JSON in prose or ```json fences despite explicit
    instructions. Extracting the outermost braces is far cheaper than a retry.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


class OpenAIProvider:
    def __init__(self, settings: Settings, name: str = "openai") -> None:
        # Imported lazily so the package is optional for mock-only runs.
        from openai import AsyncOpenAI

        base_url = (settings.llm_base_url or "").strip() or None
        api_key = (settings.llm_api_key or settings.openai_api_key or "").strip()

        # Hosted OpenAI genuinely needs a key. A self-hosted vLLM/NIM endpoint
        # often does not, so only hard-fail when we're pointed at OpenAI itself.
        if not api_key and base_url is None:
            raise ProviderError("OPENAI_API_KEY missing")

        # Gateways vary in where they look for the credential. Send the real key
        # on the configured header AND as the standard Bearer token, so we work
        # whichever one the server reads. Set llm_disable_bearer when a gateway
        # actively rejects an Authorization header it doesn't recognise.
        headers: dict[str, str] = {}
        auth_header = (settings.llm_api_key_header or "").strip()
        if auth_header and api_key:
            headers[auth_header] = api_key

        self._extra_headers: dict[str, Any] = {}
        if settings.llm_disable_bearer and auth_header.lower() != "authorization":
            from openai._types import Omit

            # Must be per-request: the SDK only recognises Omit in request-level
            # headers, and silently keeps the Bearer if passed via default_headers.
            # Skipped when the configured header IS Authorization — there the
            # custom value has already replaced the Bearer token, and omitting it
            # would strip the credential entirely and 401.
            self._extra_headers["Authorization"] = Omit()

        self.name = name
        self.model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._timeout = settings.llm_timeout_s
        self._strip_reasoning = settings.llm_strip_reasoning
        self._use_json_mode = settings.llm_use_json_mode
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=settings.llm_timeout_s,
            default_headers=headers or None,
        )

    async def generate_text(
        self, *, system: str, user: str, max_tokens: int = 512
    ) -> LLMResult:
        try:
            resp = await self._client.chat.completions.create(
                extra_headers=self._extra_headers or None,
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
            text=_message_text(resp.choices[0].message, self._strip_reasoning),
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
        kwargs: dict[str, Any] = {}
        if self._use_json_mode:
            # Not every OpenAI-compatible server implements JSON mode; the schema
            # is also stated in the prompt above, so this is belt-and-braces.
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(
                extra_headers=self._extra_headers or None,
                model=self.model,
                temperature=self._temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except Exception as exc:
            raise _provider_error(exc) from exc
        content = _json_payload(
            _message_text(resp.choices[0].message, self._strip_reasoning)
        ) or "{}"
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
                extra_headers=self._extra_headers or None,
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
            tool_calls=calls, text=_message_text(msg, self._strip_reasoning),
            model=self.model, provider=self.name, usage=usage,
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


LLM_ROLES = ("agents", "mitigation", "notification")

# Global settings a role may override, in the order they are resolved.
_ROLE_OVERRIDABLE = ("llm_model", "llm_base_url", "llm_api_key", "llm_api_key_header")


def settings_for_role(settings: Settings, role: str | None) -> Settings:
    """Settings with any per-role LLM overrides applied.

    Returns the *same object* when the role configures nothing, which callers
    rely on to detect "no override" cheaply via identity.
    """
    if not role:
        return settings
    if role not in LLM_ROLES:
        raise ProviderError(f"unknown LLM role {role!r}; expected one of {LLM_ROLES}")
    updates = {
        field: value
        for field in _ROLE_OVERRIDABLE
        if (value := getattr(settings, f"{field}_{role}", ""))
    }
    return settings.model_copy(update=updates) if updates else settings


def role_provider(settings: Settings, role: str, default: LLMProvider) -> LLMProvider:
    """Provider for one role, or `default` when the role overrides nothing.

    Falling back to the injected provider (rather than rebuilding an equivalent
    one) keeps dependency injection intact: tests and callers that hand in a
    specific provider still have it used everywhere unless an override is
    explicitly configured.
    """
    scoped = settings_for_role(settings, role)
    if scoped is settings:
        return default
    return build_provider(scoped)


def build_provider(settings: Settings, role: str | None = None) -> LLMProvider:
    settings = settings_for_role(settings, role)
    match settings.llm_provider:
        case "openai":
            return OpenAIProvider(settings)
        case "nemotron":
            # Any OpenAI-compatible endpoint: vLLM, NIM, Ollama, or a routing
            # gateway. Same wire protocol, so OpenAIProvider drives it directly;
            # the distinct name keeps logs and metrics honest about what ran.
            if not settings.llm_base_url:
                raise ProviderError("LLM_BASE_URL required for provider=nemotron")
            return OpenAIProvider(settings, name="nemotron")
        case "anthropic":
            return AnthropicProvider(settings)
        case "gemini":
            return GeminiProvider(settings)
        case "mock":
            return MockProvider(settings)
        case _:
            raise ProviderError(f"unknown provider {settings.llm_provider}")