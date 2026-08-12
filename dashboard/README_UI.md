# Agentic Operations Dashboard (Streamlit)

A role-aware dashboard that drives the **autonomous agents**. It is a thin client
of the FastAPI backend — it re-implements no logic. Its primary action calls
`POST /v1/agentic/run`, which launches two ReAct tool-calling agents in parallel
(a per-order **triage agent** and an aggregate **root-cause agent**), then renders
their activity, the ranked root causes, escalations, and the human-approval queue.

## Tabs

- **Overview** — KPIs (at-risk orders, notifications drafted, approvals queued,
  validated root causes) + each agent's tool-call count + escalation status.
- **Agents** — the two agents side by side: tool-call count, the ordered
  trajectory of tools each one called, and its final answer.
- **Root causes** — ranked validated causes with mitigation / expected effect.
- **Approvals** — Manager only: review the exact drafted message per order and
  approve/reject (`POST /v1/agentic/approvals/{id}`); the queue shrinks as you act.

## Roles (enforced by the API's JWT scopes)

| Capability | Analyst | Operations Manager |
|---|---|---|
| Run the agents (`/v1/agentic/run`) | ✅ | ✅ |
| Overview, Agents, Root causes | ✅ | ✅ |
| Review & approve customer messages | 🔒 (403) | ✅ |

The Approvals workspace needs `notify:send`, which only `operations_manager`
(and `admin`) carry. An Analyst sees the queue size but not the message bodies.

## Run it (two terminals)

```bash
pip install -r requirements.txt              # API deps (from project root)
pip install -r dashboard/requirements.txt    # dashboard deps
```

```bash
# Terminal 1 — API
uvicorn app.api.app:create_app --factory --reload --port 8000
# Terminal 2 — dashboard
streamlit run dashboard/streamlit_app.py
```

Then: set the API URL, choose **Analyst** or **Operations Manager**, Connect,
upload a batch, and click **Run agents**. Reconnect as the other role to show the
difference (the Approvals tab unlocks for a Manager).

## Seeing the agents actually call tools

With `LLM_PROVIDER=mock` (no API keys) the agents don't call tools, so the
Agents tab shows empty trajectories — but coverage, root causes, and approvals
are still produced by the deterministic guarantees, so the rest of the dashboard
is fully populated. To see real ReAct trajectories (tool calls per agent), point
the API at a real provider (e.g. `LLM_PROVIDER=openai` with `OPENAI_API_KEY`).

## Notes

- The approval store is in-memory in the starter, so create and act on the queue
  within the same running API process.