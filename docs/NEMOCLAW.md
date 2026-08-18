# NemoClaw integration

Three steps, in dependency order. Step 1 stands alone and is worth doing whether
or not you continue. Step 2 is useful to any MCP client. Step 3 is optional and
only pays off if you want a conversational surface.

---

## Step 1 — Local Nemotron inference

`OpenAIProvider` already drives any OpenAI-compatible server, so this is
configuration, not code. Add to `.env`:

```ini
LLM_PROVIDER=nemotron
LLM_BASE_URL=http://localhost:8000/v1     # the /v1 suffix is required
LLM_API_KEY=
LLM_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
LLM_STRIP_REASONING=true
LLM_USE_JSON_MODE=false
```

Notes that will save you a debugging session:

- `LLM_PROVIDER=nemotron` **fails fast** if `LLM_BASE_URL` is unset. That is
  deliberate — it makes "I meant local" explicit rather than silently falling
  back to `api.openai.com`.
- Leave `LLM_STRIP_REASONING=true` unless vLLM runs with
  `--reasoning-parser nemotron_v3`. Without a parser the `<think>` trace lands
  in `message.content` and every `json.loads` downstream fails.
- Start with `LLM_USE_JSON_MODE=false`. Not every vLLM build implements
  `response_format={"type":"json_object"}`; the schema is stated in the prompt
  regardless, and `_json_payload()` extracts the object from surrounding prose.
  Turn it on once you have confirmed the build supports it.

Verify with `python scripts/check_nemotron.py` before moving on.

Per-role routing is where this design earns its keep. Notification drafting is
short customer-facing text with no reasoning requirement; mitigation narratives
go to ops leadership at roughly two calls per run. On a single GPU, leave the
role overrides blank and share one model. With headroom:

```ini
LLM_MODEL_NOTIFICATION=nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
LLM_MODEL_MITIGATION=nvidia/llama-3.3-nemotron-super-49b-v1.5
```

---

## Step 2 — MCP server over `ToolRegistry`

### What is published

| Tool | Source | Notes |
|---|---|---|
| `summarize_data` | `ToolRegistry` | takes `dataset` |
| `segment_late_rate` | `ToolRegistry` | takes `dataset` |
| `run_root_cause_analysis` | `ToolRegistry` | takes `dataset` |
| `score_triage_queue` | `ToolRegistry` | takes `dataset` |
| `list_datasets` | new | batch files, newest first |
| `get_recent_runs` | `RunStore` | completed run summaries |
| `get_run_findings` | `RunStore` | full result, trajectories stripped |

**Withheld:** `propose_customer_notification`, `propose_ops_escalation`.

The exclusion is structural — `bridge.exposed_tools()` filters on
`requires_approval`, so any action tool added later is excluded automatically.
Publishing one would require clearing the flag that gates it everywhere else,
which is a change you would notice in review.

This is the design decision worth understanding: the approval gate lives outside
the MCP boundary. A sandboxed agent proposing a notification would enqueue work
it cannot execute, so publishing the proposal tools buys nothing except a way
for a chat client to fill the human review queue. Proposals stay on the
authenticated HTTP path, where a principal and a run_id stand behind them.

### Configuration

```ini
MCP_ENABLED=true
MCP_BEARER_TOKEN=            # openssl rand -hex 32
MCP_DATASET_DIR=./data/archive
MCP_ALLOWED_HOSTS=shipment-mcp.example.com
MCP_CONTEXT_CACHE_SIZE=4
```

`MCP_ENABLED` defaults to false and startup **refuses to boot** if it is true
with an empty token — a public analytics endpoint should not be reachable
through a config typo.

`MCP_ALLOWED_HOSTS` is required behind a TLS terminator. The SDK validates the
`Host` header for DNS-rebinding protection, and Caddy rewrites it to the public
name, which this process would otherwise reject.

`MCP_DATASET_DIR` defaults to `TRIGGER_ARCHIVE_DIR`. Point it at the archive so
the conversational surface sees exactly the batches the pipeline already
processed, and nothing else on the host. Dataset names are resolved against that
one root; anything containing a path separator is rejected before resolution,
and symlinks are judged on their real target.

### Running it

```bash
python -m uvicorn app.api.app:app --port 8000       # MCP mounts at /mcp
caddy run --config deploy/Caddyfile                 # TLS on :443
```

Smoke test:

```bash
curl -s -X POST https://shipment-mcp.example.com/mcp/ \
  -H "Authorization: Bearer $MCP_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
       "protocolVersion":"2025-06-18","capabilities":{},
       "clientInfo":{"name":"probe","version":"1"}}}'
```

A 401 without the header and a `serverInfo` block with it means you are done.

---

## Step 3 — Register with NemoClaw

Prerequisites, in the order they will bite:

1. **A Brev VM, not a container.** NemoClaw needs Docker daemon access; nested
   Docker in a Brev container will not work.
2. **NVIDIA Container Toolkit and a healthy CDI spec.** Generic Linux GPU hosts
   also need `NEMOCLAW_EXPERIMENTAL=1` or `NEMOCLAW_PROVIDER=install-vllm`.
3. **A public DNS name with a real certificate.** `mcp add` rejects `http://`
   and local URLs outright. This is the step that most often stalls.

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
nemoclaw onboard

nemoclaw <sandbox> mcp add shipment \
  --url https://shipment-mcp.example.com/mcp/ \
  --env SHIPMENT_MCP_TOKEN=<the MCP_BEARER_TOKEN value>
```

Check the current flag names with `nemoclaw <sandbox> mcp add --help` — the CLI
is alpha and the surface moves.

Constraints worth knowing before you hit them:

- Exactly one `--env` bearer credential per server, and the name must be
  distinct per server in the same sandbox. `NEMOCLAW_*`, `OPENCLAW_*`,
  `OPENSHELL_*`, `PATH`, `NODE_OPTIONS` and similar are rejected.
- The full URL including path is persisted **and displayed**. Never put the
  credential in the URL.
- NemoClaw stores only the variable name; OpenShell keeps the raw value on the
  host and resolves it at egress, so the token never enters the sandbox.

---

## Operational notes

**The token is a single shared credential.** Anyone holding it can read all
analytics. Rotate it by updating `.env`, restarting, and re-running `mcp add`.
Only the MCP mount is proxied publicly; the REST API, `/metrics` and the
dashboard stay on the private interface.

**First call on a dataset is slow.** It runs `ingest()` over the whole frame.
Contexts are cached and keyed on file mtime, so an updated batch invalidates its
own entry. `MCP_CONTEXT_CACHE_SIZE` bounds memory — each entry holds the full
frame, so 4 batches of 180k rows is real RSS. Size it against the box.

**Trajectories are stripped** from `get_run_findings`. They are large,
repetitive, and would consume a conversational agent's context window. Use the
HTTP `/v1/runs/{id}` endpoint when you need the full trace for debugging.

**The server instructions carry your data-integrity stance** — evidence grades,
the no-carrier-field constraint, and "do not recompute the numbers." A remote
model cannot read `DESIGN.md`, so anything it must not do has to be stated in
`server.INSTRUCTIONS`. Extend it there when the stance changes.