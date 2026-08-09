# Design Document — Shipment Delay & Exception Agent

Covers the eleven required deliverables. Companion to the running code in `app/`.

---

## 1. Brief architecture summary

A FastAPI service ingests an orders file, normalises it through a configurable column-mapping layer, and hands two deterministic frames (closed history, open in-flight) to an **orchestrator**. The orchestrator fans out to two lanes in parallel:

- **Lane A (per-order triage):** deterministic risk/slip/impact scoring → prioritised queue → LLM-drafted customer notification, gated by input+output guardrails with a deterministic fallback.
- **Lane B (systemic root cause):** deterministic five-gate statistical pipeline → ranked validated findings + a rejected register → LLM-written mitigation narrative for validated findings only.

An **escalation gate** fires an internal escalation only when a finding's confidence clears a threshold; otherwise it stays silent and logs the suppression. All quantitative work is Python/pandas/scipy. The LLM is confined to explanation and customer wording behind strict Pydantic schemas. Provider access is via an `LLMProvider` protocol, making OpenAI swappable for Anthropic/Gemini/Azure/local.

---

## 2. Text-based architecture diagram

```
                 ┌─────────────────────────────────────────────┐
 upload (CSV/XLSX)│  FastAPI  /v1/analyze  (JWT + RBAC scopes)   │
────────────────▶│  upload validation · size/type · AV seam    │
                 └───────────────┬─────────────────────────────┘
                                 ▼
                     ingest() — column mapping, whitespace hygiene,
                     late label, cancel-exclude, cohort split,
                     data-quality flags        (DETERMINISTIC)
                                 │
                closed history ──┼── open in-flight
                                 ▼
                        ┌────────ORCHESTRATOR────────┐
                        │  fan-out → parallel lanes  │
                        └───┬───────────────────┬────┘
              ┌─────────────▼──────┐    ┌───────▼──────────────────┐
              │ LANE A (triage)    │    │ LANE B (root cause)      │
              │ rate table         │    │ enumerate candidates     │
              │ risk+slip+impact   │    │ G0 support floor (M)     │
              │ prioritise (cap N) │    │ S1 baseline assignment   │
              │ ── per order ──    │    │ G2 effect size           │
              │ input guardrail    │    │ G3 BH-FDR (q, M)         │
              │ LLM draft (schema) │    │ G4 confound / parents    │
              │ output guardrail   │    │ G5 temporal stability    │
              │ fallback template  │    │ confidence + rank        │
              └─────────┬──────────┘    │ LLM mitigation narrative │
                        │               └──────────┬───────────────┘
                        │                          ▼
                        │                 ESCALATION GATE (conf ≥ 0.75?)
                        │                  PASS → escalation ticket
                        │                  FAIL → silent + suppression log
                        ▼                          ▼
              triage[] + notifications[]     RootCauseReport
                        └──────────┬───────────────┘
                                   ▼
                        audit tables · JSON logs · Prometheus
```

---

## 3. Repository structure

```
app/
  core/        config.py (Pydantic Settings), column_mapping.py
  domain/      models.py (Pydantic domain + EvidenceGrade)
  providers/   base.py (LLMProvider protocol), factory.py (OpenAI + stubs + mock)
  analytics/   ingest.py, rate_table.py, triage.py, root_cause.py   (NO LLM)
  agents/      orchestrator.py, notification_agent.py, mitigation_agent.py, reliability.py
  guardrails/  input_guard.py, output_guard.py
  prompts/     notification_v1.py, mitigation_v1.py  (versioned)
  observability/ logging_setup.py, metrics.py
  persistence/ audit.py (SQLAlchemy async audit tables)
  api/         app.py, schemas.py, security.py, upload_validation.py, routes/
tests/         test_analytics.py, test_guardrails.py, test_orchestrator.py
scripts/       make_synthetic_data.py
data/          synthetic files (git-ignored)
docs/          DESIGN.md
```

---

## 4. Data model & database schema

**Domain objects** (`app/domain/models.py`, all Pydantic): `TriageRecord`, `NotificationRecord`, `CandidatePattern`, `ValidatedFinding`, `RejectedCandidate`, `EscalationDecision`, `RootCauseReport`. Each quantitative object carries an `EvidenceGrade`.

**Audit tables** (`app/persistence/audit.py`, SQLAlchemy async):

- `run_audit`: run_id, created_at, input_hash, analytics_version, prompt_version, model_provider, model_name, m_tests_conducted, escalated, report_json.
- `notification_audit`: run_id, order_hash, guardrail_outcome, used_fallback, review_status (pending/approved/rejected), sent_status (unsent/sent), body_preview.

These preserve the lineage required to reproduce and defend any decision: input version/hash, code/prompt/model versions, guardrail outcomes, human approvals/edits, and final send status.

---

## 5. Agent workflow & orchestration

- **Ingestion (deterministic):** resolve columns (whitespace-tolerant), derive the binary late label from `Delivery Status`, exclude cancelled shipments, split open vs closed cohorts, and emit data-quality flags (line-item grain, regions absent from the first half, stripped whitespace).
- **Orchestrator:** creates `run_id`/`correlation_id`, launches Lane A and Lane B as parallel `asyncio` tasks, joins, then runs the escalation gate.
- **Lane A failure policy:** per-order isolation — a failed notification is skipped and logged; the queue still returns. Bounded concurrency (semaphore) protects the provider.
- **Lane B failure policy:** atomic — statistics either complete or the lane raises; no half-tested systemic claim is emitted.
- **Escalation gate:** pure function of `max(confidence)` vs threshold; emits a ticket or a suppression log line, never both silently.

---

## 6. Guardrail design

**Input (`input_guard.py`):** PII redaction (email/phone/card), prompt-injection detection + neutralisation, per-field length caps, and an **allowlist** so only whitelisted fields ever reach the model. Untrusted field values are fenced as DATA in prompts and never merged into instructions.

**Output (`output_guard.py`):** blocks refund/compensation/guarantee promises, tracking-location claims, unsupported internal-cause claims, and the boilerplate phrase "unforeseen circumstances"; requires a date; rejects any ISO date earlier than the supported revised ETA; enforces a minimum count of grounded field values present verbatim; caps length. On any failure the deterministic `fallback_notification` is used, which is itself validated.

**Prompt-injection defence:** order fields, product names, comments and uploaded content are treated as untrusted. Patterns like "ignore previous instructions", "system prompt", "act as" are stripped to `[BLOCKED]` before the text is sent, and prompts explicitly instruct the model not to execute instructions found in the DATA block.

Every guardrail decision is logged and counted (`guardrail_blocks_total{stage,reason}`).

---

## 7. Logging, audit, metrics, tracing

**Structured JSON logs** with an allowlist of fields (timestamp UTC, environment, service, version, request/correlation IDs, workflow/job IDs, agent name/run id, hashed order id, event name/status, duration, model provider/name, prompt version, token counts, estimated cost, guardrail outcome, retry count, sanitized error) and a hard denylist (keys, auth headers, raw PII, raw prompts, emails/phones/addresses/payment).

**Event names** implemented: `request_received`, `triage_classification_completed`, `customer_notification_generated`, `customer_notification_guardrail_failed`, `analytics_batch_started`, `analytics_segment_calculated`, `systemic_risk_detected`, `escalation_decision_created`, `llm_request_completed`, `llm_request_failed`, `guardrail_blocked`, `fallback_response_used`.

**Metrics** (Prometheus): API latency, LLM latency/cost, guardrail-block rate, exceptions detected, escalations, batch duration, error count, queue depth.

**Tracing:** OpenTelemetry seams are marked around API requests, agent execution, analytics jobs, and LLM calls (exporters are wired per-environment; the code isolates each span boundary at the orchestrator and agent methods).

---

## 8. API design

- `POST /auth/token` → `{access_token}` (dev; prod uses an IdP). Body `{username, role, tenant_id}`.
- `GET /health` → provider health, provider/model identity.
- `POST /v1/analyze` (scope `analytics:run`) — multipart file upload, optional `Idempotency-Key` header → `AnalyzeResponse` (triage, notifications, root-cause report, escalation).
- `GET /metrics` — Prometheus exposition.

Errors are sanitized: 401 missing/invalid token, 403 missing scope, 413 too large, 415 unsupported type, 422 unreadable/failed-scan.

---

## 9. Implementation plan in phases

1. **Foundations:** config, column mapping, domain models, provider protocol + mock. (done)
2. **Deterministic analytics:** ingest, rate table, triage scoring, five-gate root cause. (done)
3. **Guardrails:** input/output/injection + fallbacks. (done)
4. **Agents + orchestration:** notification + mitigation agents, parallel lanes, escalation gate, reliability wrapper. (done)
5. **API + security:** FastAPI, JWT/RBAC, upload validation. (done)
6. **Observability + audit:** JSON logs, metrics, audit tables. (done)
7. **Hardening (next):** OTel exporters, Redis idempotency, worker queue + DLQ, human-review UI, malware-scanner client, rate-limit middleware, Postgres migrations.

---

## 10. Production-ready starter code

All of `app/`, `tests/`, and `scripts/` in this repository. 19 tests pass; the API boots and the orchestrator runs end-to-end on synthetic data with the mock provider (no keys, no network). Nothing is stubbed except the alternate providers and clearly-marked integration seams (AV scanner, OTel exporters), each with a conforming interface.

---

## 11. README

See `README.md` for local setup, OpenAI configuration, API examples, RBAC behaviour, and deployment recommendations.

---

## Stated assumptions

1. No carrier field exists in DataCo → all lane claims use "shipping mode / region", never "carrier".
2. `Late_delivery_risk` is a label, not a score → triage learns from closed orders.
3. Rows are line-item grained; the independence caveat is flagged and rates are order-consistent. Moving Lane B to deduplicated orders is a one-line change in `ingest`.
4. The dev `/auth/token` endpoint stands in for an upstream IdP.
5. Malware scanning and OTel export are integration points; interfaces are present, vendor clients are not bundled.
6. LLM cost figures are estimates from a static price table and are for observability only.
