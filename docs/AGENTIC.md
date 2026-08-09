# Agentic Layer — Autonomous Tool-Calling Agents

This layer sits alongside the deterministic pipeline (`app/agents/orchestrator.py`).
It makes the system **agentic in the autonomous ReAct sense**: the LLM reasons and
chooses its own tools in a loop, rather than following a fixed sequence.

## The core safety idea

The LLM decides **which** tool to call and with what arguments. But every tool
body is **deterministic Python** from the existing analytics layer. The model
never computes a metric — it orchestrates calls to functions that do. This keeps
a free-reasoning agent fully auditable and prevents fabricated logistics facts.

```
LLM (reasoning, tool selection)  →  Tool (deterministic pandas/scipy)  →  observation
        ▲                                                                     │
        └─────────────────── loop until final answer ────────────────────────┘
```

## Two autonomous agents

Both run in parallel under `AgenticOrchestrator`, each with its own tool subset:

| Agent | Allowed tools | Goal |
|---|---|---|
| Triage agent | summarize_data, segment_late_rate, score_triage_queue, **propose_customer_notification** | Find highest-impact at-risk orders, propose notifications |
| Root-cause agent | summarize_data, segment_late_rate, run_root_cause_analysis, **propose_ops_escalation** | Validate a systemic cause, propose an escalation |

## Tools

| Tool | Deterministic body | Approval? |
|---|---|---|
| summarize_data | ingest summary | no |
| segment_late_rate | groupby rate | no |
| score_triage_queue | `analytics/triage.py` | no |
| run_root_cause_analysis | five-gate `analytics/root_cause.py` | no |
| propose_customer_notification | records pending action | **yes** |
| propose_ops_escalation | records pending action | **yes** |

Analysis tools are free to call. The two `propose_*` tools are **side-effecting**;
the agent can only queue them. Nothing is sent or filed without a human decision.

## Human-in-the-loop gate

1. An agent calls a `propose_*` tool.
2. The ReAct loop detects `requires_approval` and records the action as *pending*
   instead of executing it; the agent is told it was queued.
3. `AgenticOrchestrator` moves pending actions into the `ApprovalStore`.
4. A human (role with `notify:send`) lists and approves/rejects via the API.
5. **On approval only**, the real side effect executes (email client / escalation
   system — wired at `agentic_routes.decide_approval`).

## Automation triggers

`app/triggers/automation.py` — all three converge on `dispatch_run()`:

- **Scheduled:** `run_scheduler()` scans an inbox every N seconds (cron in prod).
- **File-drop:** `watch_inbox()` polls a directory for new CSV/XLSX (watchdog/inotify in prod).
- **Webhook:** `POST /v1/agentic/webhook` (S3 key reference in prod).

Each source file is hashed; a previously seen hash is skipped (idempotent).

## New API endpoints

| Method | Path | Scope | Purpose |
|---|---|---|---|
| POST | /v1/agentic/run | analytics:run | Launch both autonomous agents on an upload |
| POST | /v1/agentic/webhook | analytics:run | Event-driven trigger |
| GET | /v1/agentic/approvals | notify:send | List pending proposed actions |
| POST | /v1/agentic/approvals/{id} | notify:send | Approve/reject; approval executes the action |

## Provider support

Tool-calling is added to the `LLMProvider` protocol as `generate_with_tools`.
`OpenAIProvider` implements it against the Chat Completions tools API.
`MockProvider` implements a scripted planner (reads a `<<PLAN>>` block) so the
whole ReAct loop is testable with no network. Anthropic/Gemini stubs need the
same method filled in with their tool-use APIs — no orchestrator changes.

## When to use which orchestrator

- **`agents/orchestrator.py`** (deterministic, fixed pipeline): predictable,
  cheapest, best for the guaranteed-coverage batch. Every order scored, every
  segment tested.
- **`agentic/orchestrator.py`** (autonomous ReAct): flexible investigation, good
  when you want the agent to decide what to look at and explain its path. Costs
  more LLM calls and is non-deterministic in trajectory.

Both share the same tools and the same approval gate, so their outputs are equally
auditable.
