# Shipment Delay & Exception Agent

Enterprise multi-agent system that (1) triages individual at-risk shipments and drafts grounded customer notifications, and (2) identifies **systemic** delay root causes with a five-gate statistical pipeline, ranks them, and produces mitigation recommendations for ops leadership.

Every numeric claim is computed deterministically in Python/pandas/scipy. The LLM only explains validated metrics and drafts customer messages — it never computes a rate, rank, or p-value. The architecture is LLM-provider-agnostic: switch OpenAI → Anthropic/Gemini/Azure/local by changing one config value.

## Data-integrity stance

- The DataCo export has **no carrier field**. The system therefore never emits a "carrier" claim; it uses **shipping lane** / **shipping mode–region combination**.
- Every output is tagged with an **evidence grade**: `observed_fact`, `data_supported_risk`, or `hypothesis`.
- `Late_delivery_risk` in the source equals the late label exactly — it is treated as a **label**, not a predictive score. Triage predictions are learned from closed orders and applied to open ones.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # leave keys blank to run with the mock provider

# generate synthetic, PII-free data
python scripts/make_synthetic_data.py --rows 8000 --out data/synthetic_orders.csv

# run tests (no network, uses the mock provider)
pytest -q

# start the API
uvicorn app.api.app:app --reload
```

## OpenAI configuration

Set in `.env`:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Switching providers later is a one-line change (`LLM_PROVIDER=anthropic`). The `AnthropicProvider` / `GeminiProvider` stubs already conform to the `LLMProvider` protocol; fill in the SDK calls and no business code changes.

## API examples

Issue a scoped dev token (production uses an upstream IdP):

```bash
curl -s localhost:8000/auth/token -H 'content-type: application/json' \
  -d '{"username":"amy","role":"analyst"}'
# -> {"access_token":"<jwt>","token_type":"bearer"}
```

Analyze an orders file:

```bash
curl -s localhost:8000/v1/analyze \
  -H "Authorization: Bearer <jwt>" \
  -H "Idempotency-Key: run-2026-08-08-001" \
  -F "file=@data/synthetic_orders.csv"
```

Response (abridged):

```json
{
  "run_id": "e456934f...",
  "triage": [{"order_id":"12","p_late":0.80,"revised_eta":"2026-08-14",
              "reason_code":"mode_risk","impact_score":420.5,
              "evidence_grade":"data_supported_risk"}],
  "notifications": [{"order_id":"12","subject":"...","body":"...",
                     "remedy_tier":2,"validator_pass":true,"used_fallback":false}],
  "report": {
    "candidates_enumerated": 1054, "m_tests_conducted": 314,
    "global_late_rate": 0.582,
    "findings": [{"label":"shipping_mode=First Class","seg_rate":1.0,
                  "lift":1.72,"excess_orders":501,"confidence":0.74,
                  "evidence_grade":"data_supported_risk","mitigation":"..."}],
    "rejected": [{"pattern_id":"...","failed_gate":"gate4_confound"}],
    "escalation": {"escalated": false, "confidence": 0.74, "threshold": 0.75,
                   "suppression_reason": "best confidence 0.74 < 0.75"}
  }
}
```

RBAC: `viewer` → 403 on `/v1/analyze`; missing token → 401.

Health and metrics: `GET /health`, `GET /metrics` (Prometheus).

## Roles

`admin`, `operations_manager`, `support_agent`, `analyst`, `viewer` — mapped to capability scopes in `app/api/security.py`.

## Deployment recommendations

- **Runtime:** container the app; run `uvicorn`/`gunicorn` behind an ingress that terminates TLS (encryption in transit).
- **State:** replace the in-memory idempotency cache with Redis; point `DATABASE_URL` at managed Postgres with encryption-at-rest enabled.
- **Async batch:** move Lane B onto a worker queue (Celery/Arq) with a dead-letter queue; the `/v1/analyze` endpoint can enqueue and return a `job_id`.
- **Secrets:** inject via the platform secret manager; never bake into images. `.env.example` only.
- **Observability:** scrape `/metrics`; ship JSON logs to your aggregator; wire OpenTelemetry exporters (seams noted in `docs/DESIGN.md`).
- **Retention:** apply TTLs on audit/PII tables per policy; hash order IDs in logs (already enforced).
- **Load:** for 180k orders, Lane B runs in-process in seconds; parallelise notification drafting with bounded concurrency (already implemented) and rate-limit the provider.

See `docs/DESIGN.md` for the full architecture, schema, workflow, guardrail, and phasing detail.
