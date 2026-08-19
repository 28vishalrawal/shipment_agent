"""End-to-end smoke test for the MCP surface.

Runs a real MCP client against the running server — the same protocol a
NemoClaw sandbox uses — so the whole chain is verified locally, before any
tunnel or certificate is involved. If this passes, the only thing standing
between you and `mcp add` is a public HTTPS hostname.

    python scripts/mcp_smoke.py                        # localhost:8000
    python scripts/mcp_smoke.py --url https://host/mcp/

The token is read from MCP_BEARER_TOKEN, or from .env if that is unset, so it
never has to be typed on the command line (where it would land in shell
history).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    import httpx2 as httpx_mod
except ImportError:  # older SDK builds ship plain httpx
    import httpx as httpx_mod

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

# Tools that must never appear. Publishing either would put a real-world side
# effect behind a single shared bearer token, outside the human approval gate.
FORBIDDEN = {"propose_customer_notification", "propose_ops_escalation"}
EXPECTED = {
    "summarize_data", "segment_late_rate", "run_root_cause_analysis",
    "score_triage_queue", "list_datasets", "get_recent_runs", "get_run_findings",
}


def load_token() -> str:
    token = os.environ.get("MCP_BEARER_TOKEN", "").strip()
    if token:
        return token
    env = Path(".env")
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("MCP_BEARER_TOKEN="):
                return line.split("=", 1)[1].strip()
    return ""


def text_of(result) -> str:
    for block in result.content:
        if getattr(block, "text", None):
            return block.text
    return ""


async def main(url: str, token: str) -> int:
    failures = 0
    client = httpx_mod.AsyncClient(headers={"Authorization": f"Bearer {token}"})

    try:
        async with streamable_http_client(url, http_client=client) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                info = await session.initialize()
                print(f"{PASS} connected to {info.server_info.name} "
                      f"v{info.server_info.version}")

                listing = await session.list_tools()
                names = {t.name for t in listing.tools}
                print(f"\n  tools: {', '.join(sorted(names))}\n")

                leaked = names & FORBIDDEN
                if leaked:
                    print(f"{FAIL} approval-gated tools exposed: {leaked}")
                    failures += 1
                else:
                    print(f"{PASS} approval-gated tools withheld")

                missing = EXPECTED - names
                if missing:
                    print(f"{FAIL} expected tools absent: {missing}")
                    failures += 1
                else:
                    print(f"{PASS} all expected tools present")

                # --- datasets -------------------------------------------------
                raw = text_of(await session.call_tool("list_datasets", {}))
                datasets = json.loads(raw).get("datasets", [])
                if not datasets:
                    print(f"{WARN} no datasets found — put a .csv in MCP_DATASET_DIR")
                    print("       (dataset-backed checks skipped)")
                else:
                    print(f"{PASS} list_datasets returned {len(datasets)}: "
                          f"{datasets[0]['name']}")

                    raw = text_of(await session.call_tool(
                        "summarize_data", {"dataset": "latest"}))
                    body = json.loads(raw)
                    if "error" in body:
                        print(f"{FAIL} summarize_data: {body['error']}")
                        failures += 1
                    else:
                        r = body["result"]
                        print(f"{PASS} summarize_data: {r['input_rows']} rows, "
                              f"late rate {r['global_late_rate']}")

                    # Containment: an untrusted name must not escape the root.
                    raw = text_of(await session.call_tool(
                        "segment_late_rate",
                        {"dataset": "../../etc/passwd",
                         "dimension": "shipping_mode", "value": "x"}))
                    if "error" in json.loads(raw):
                        print(f"{PASS} path traversal rejected")
                    else:
                        print(f"{FAIL} path traversal NOT rejected")
                        failures += 1

                # --- run history ----------------------------------------------
                raw = text_of(await session.call_tool(
                    "get_recent_runs", {"limit": 5}))
                runs = json.loads(raw).get("runs", [])
                print(f"{PASS} get_recent_runs returned {len(runs)}")
                if runs:
                    raw = text_of(await session.call_tool(
                        "get_run_findings", {"run_id": runs[0]["run_id"]}))
                    body = json.loads(raw)
                    trimmed = body.get("result", {}).get("root_cause_agent", {})
                    if "trajectory" in trimmed:
                        print(f"{FAIL} trajectories not stripped from findings")
                        failures += 1
                    else:
                        print(f"{PASS} get_run_findings ok, trajectories stripped")
                else:
                    print(f"{WARN} no completed runs — run the pipeline once "
                          "to exercise get_run_findings")
    except Exception as exc:
        print(f"{FAIL} connection failed: {type(exc).__name__}: {exc}")
        print("\n  check: is uvicorn running? does the token match .env?")
        return 1
    finally:
        await client.aclose()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
    else:
        print(f"{PASS} all checks passed — the MCP surface is ready")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/mcp/")
    args = ap.parse_args()

    tok = load_token()
    if not tok:
        sys.exit("MCP_BEARER_TOKEN not set and not found in .env")
    sys.exit(asyncio.run(main(args.url, tok)))