"""Capability probe for an OpenAI-compatible Nemotron endpoint.

Run this BEFORE the first full agentic run. The app degrades quietly when the
server lacks tool calling (agents return prose, the deterministic lane silently
carries the run) or JSON mode (structured output falls back to templates), so
these three checks are worth thirty seconds.

    export LLM_PROVIDER=nemotron
    export LLM_MODEL=nemotron-omni-30b-a3b-fp8
    export LLM_BASE_URL=https://.../test-router/v1
    export LLM_API_KEY=...
    export LLM_API_KEY_HEADER=x-api-key
    python scripts/check_nemotron.py
"""
from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel

sys.path.insert(0, ".")

from app.core.config import get_settings  # noqa: E402
from app.providers.factory import LLM_ROLES, build_provider  # noqa: E402
from app.tools.registry import build_registry  # noqa: E402


class Mitigation(BaseModel):
    action: str
    rationale: str


async def main() -> int:
    s = get_settings()
    roles = {r: build_provider(s, r) for r in LLM_ROLES}

    def endpoint(p) -> str:
        # MockProvider and other non-HTTP providers have no client.
        client = getattr(p, "_client", None)
        return str(getattr(client, "base_url", "n/a"))

    print("resolved roles:")
    for r, rp in roles.items():
        print(f"  {r:13} model={rp.model:30} {endpoint(rp)}")
    print()

    # Roles sharing an identical model + endpoint only need probing once.
    seen: dict[tuple, str] = {}
    unique: list[tuple[str, object]] = []
    for r, rp in roles.items():
        key = (rp.model, endpoint(rp))
        if key in seen:
            print(f"note: '{r}' shares {seen[key]}'s endpoint and model — probed once\n")
            continue
        seen[key] = r
        unique.append((r, rp))

    failures: list[str] = []
    for role, p in unique:
        failures += await probe(role, p, s)

    print()
    for f in failures:
        print(f"  ! {f}")
    if not failures:
        print("All checks passed — safe to run the agentic pipeline.")
    return 1 if failures else 0


async def probe(role: str, p, s) -> list[str]:
    print(f"--- role: {role} ({p.model}) ---")
    failures: list[str] = []

    print("[1/4] reachable ...", end=" ", flush=True)
    try:
        print("ok" if await p.health_check() else "models list empty (may be fine)")
    except Exception as exc:
        print(f"FAIL {exc}")
        return [f"{role}: endpoint unreachable ({exc})"]

    print("[2/4] chat ...", end=" ", flush=True)
    try:
        r = await p.generate_text(system="Reply with one word.", user="Hello", max_tokens=256)
        print(f"ok — {r.text[:60]!r} ({r.usage.completion_tokens} tok)")
        if "<think" in (r.text or "").lower():
            failures.append(f"{role}: reasoning trace leaked into content despite stripping")
    except Exception as exc:
        print(f"FAIL {exc}")
        return [f"{role}: chat failed ({exc}) — check model name and credentials"]

    print("[3/4] tool calling ...", end=" ", flush=True)
    if role != "agents":
        # Only the ReAct agents call tools; mitigation and notification use
        # structured output, so a model without tool support is fine for them.
        print("skipped (not required for this role)")
    else:
        try:
            reg = build_registry()
            res = await p.generate_with_tools(
                messages=[
                    {"role": "system", "content": "Use a tool to answer. Do not guess."},
                    {"role": "user", "content": "Summarise the current shipment dataset."},
                ],
                tools=reg.schemas(["summarize_data"]),
                max_tokens=512,
            )
            if res.tool_calls:
                print(f"ok — called {[c.name for c in res.tool_calls]}")
            else:
                print("NO TOOL CALLS")
                failures.append(
                    f"{role}: tool calling unavailable — start vLLM with "
                    "--enable-auto-tool-choice --tool-call-parser qwen3_coder, or the "
                    "ReAct agents cannot act"
                )
        except Exception as exc:
            print(f"FAIL {exc}")
            failures.append(f"{role}: tool calling errored: {exc}")

    print("[4/4] structured output ...", end=" ", flush=True)
    try:
        parsed, _ = await p.generate_structured_output(
            system="You propose supply-chain mitigations.",
            user="First Class shipping is 100% late. Propose one action.",
            schema=Mitigation,
            max_tokens=512,
        )
        print(f"ok — {parsed.action[:60]!r}")
    except Exception as exc:
        print(f"FAIL {exc}")
        failures.append(
            f"{role}: structured output failed ({exc}); retry with LLM_USE_JSON_MODE=false"
        )
    print()
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))