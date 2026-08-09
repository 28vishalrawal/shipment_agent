"""Routes for the agentic layer: autonomous run, webhook trigger, approvals."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.agentic.orchestrator import AgenticOrchestrator
from app.analytics.ingest import ingest
from app.api.security import Principal, require_scope
from app.api.upload_validation import read_upload
from app.approval.store import ApprovalStatus, get_approval_store
from app.core.config import get_settings
from app.observability.logging_setup import log_event, new_id
from app.providers.factory import build_provider

logger = logging.getLogger("api.agentic")
router = APIRouter(prefix="/v1/agentic", tags=["agentic"])


class ApprovalDecision(BaseModel):
    approve: bool


@router.post("/run")
async def agentic_run(
    file: UploadFile = File(...),
    principal: Principal = Depends(require_scope("analytics:run")),
) -> dict:
    """Launch the two autonomous tool-calling agents on an uploaded batch."""
    correlation_id = new_id()
    df = await read_upload(file)
    ingested = ingest(df)
    s = get_settings()
    orch = AgenticOrchestrator(build_provider(s), s)
    return await orch.run(ingested, correlation_id)


@router.post("/webhook")
async def agentic_webhook(
    file: UploadFile = File(...),
    principal: Principal = Depends(require_scope("analytics:run")),
) -> dict:
    """Event-driven trigger. In prod this would accept a reference (S3 key) and
    fetch the batch; here it accepts the file directly for a self-contained demo."""
    log_event(logger, "trigger_fired", source="webhook", tenant_id=principal.tenant_id)
    return await agentic_run(file=file, principal=principal)


@router.get("/approvals")
async def list_approvals(
    run_id: str | None = None,
    principal: Principal = Depends(require_scope("notify:send")),
) -> dict:
    store = get_approval_store()
    items = store.list_pending(run_id)
    return {
        "pending": [
            {"approval_id": i.id, "run_id": i.run_id, "type": i.action_type,
             "payload": i.payload, "status": i.status.value}
            for i in items
        ]
    }


@router.post("/approvals/{approval_id}")
async def decide_approval(
    approval_id: str,
    decision: ApprovalDecision,
    principal: Principal = Depends(require_scope("notify:send")),
) -> dict:
    store = get_approval_store()
    item = store.get(approval_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "approval not found")
    if item.status != ApprovalStatus.PENDING:
        raise HTTPException(status.HTTP_409_CONFLICT, f"already {item.status.value}")

    item = store.decide(approval_id, decision.approve, principal.sub)
    log_event(logger, "approval_decision", approval_id=approval_id,
              decision="approved" if decision.approve else "rejected",
              actor_role=principal.role)

    # On approval, THIS is where the real side effect executes. In the starter we
    # mark it executed; production wires the email/escalation client here.
    executed = False
    if decision.approve:
        store.mark_executed(approval_id)
        executed = True
        log_event(logger, "action_executed", approval_id=approval_id,
                  action_type=item.action_type)

    return {"approval_id": approval_id, "status": item.status.value, "executed": executed}
