# Operations Dashboard (Streamlit)

A role-aware dashboard for the Shipment Delay & Exception Agent. It is a **thin
client of the FastAPI backend** — it re-implements no analytics. It signs in via
`/auth/token`, uploads a batch, and renders the responses from `/v1/analyze`,
`/v1/agentic/run`, and the approvals endpoints. The Analyst vs Manager split is
enforced by the backend's JWT scopes, not just hidden in the UI.

## What each role sees

| Capability | Analyst | Operations Manager |
|---|---|---|
| Overview KPIs, escalation status | ✅ | ✅ |
| Triage queue (at-risk orders) | ✅ | ✅ |
| Root causes (ranked, with mitigation) | ✅ | ✅ |
| Drafted notifications (read-only) | ✅ | ✅ |
| **Approvals workspace** (approve/reject & send) | 🔒 hidden | ✅ |

The Approvals tab requires the `notify:send` scope, which only `operations_manager`
(and `admin`) carry — so an Analyst can review everything but cannot approve.

## Run it (two terminals)

Install deps (once):

```bash
pip install -r requirements.txt          # from the shipment_agent/ root, for the API
pip install -r dashboard/requirements.txt # for the dashboard
```

Terminal 1 — the API:

```bash
# from the shipment_agent/ project root
uvicorn app.api.app:create_app --factory --reload --port 8000
```

Terminal 2 — the dashboard:

```bash
# from the shipment_agent/ project root
streamlit run dashboard/streamlit_app.py
```

Open the Streamlit URL (usually http://localhost:8501). In the left panel:

1. Set the **API base URL** (default `http://localhost:8000`).
2. Choose **Analyst** or **Operations Manager**, then **Connect**.
3. Upload an order batch (e.g. `data/DataCoSupplyChainDataset.csv` or a synthetic file).
4. Click **Run analysis**.

To demo the role difference, connect as **Analyst** (no Approvals tab), then
reconnect as **Operations Manager** (Approvals tab appears; you can create the
queue and approve/reject individual messages).

## Notes

- The approval store in the starter is in-memory, so create and act on the queue
  within the same running API process.
- `LLM_PROVIDER=mock` on the API needs no keys and produces safe fallback drafts;
  set a real provider (e.g. OpenAI) for LLM-written messages and mitigations.
