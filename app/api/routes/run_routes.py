"""Run history: retrieve analyses produced by any trigger.

Without these, a run's root causes were visible only in the HTTP response to
whoever started it — and file-drop runs, which have no HTTP caller at all, were
visible to nobody. A manager reviewing an escalation in the approval queue needs
to reach the evidence behind it regardless of who or what triggered the run.

Read scope is analytics:read, so any authenticated user can open any run.
Provenance is recorded on the record (source + triggered_by) rather than used to
restrict access: the point is shared visibility with an auditable trail.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.security import Principal, require_scope
from app.observability.logging_setup import log_event
from app.persistence.run_store import get_run_store

logger = logging.getLogger("api.runs")
router = APIRouter(prefix="/v1/runs", tags=["runs"])


@router.get("")
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    source: str | None = Query(default=None,
                               description="upload | webhook | file_drop | scheduler"),
    principal: Principal = Depends(require_scope("analytics:read")),
) -> dict:
    """Recent runs, newest first. Summaries only — the full result (including
    agent trajectories) is large, so it is fetched per run."""
    store = get_run_store()
    runs = await asyncio.to_thread(store.list, limit, source)
    return {"count": len(runs), "runs": [r.to_dict() for r in runs]}


@router.get("/latest")
async def latest_run(
    principal: Principal = Depends(require_scope("analytics:read")),
) -> dict:
    """Most recent run. Convenient right after a file drop, when the caller has
    no run_id to look up."""
    store = get_run_store()
    record = await asyncio.to_thread(store.latest)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="no runs recorded yet")
    return record.to_dict()


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    principal: Principal = Depends(require_scope("analytics:read")),
) -> dict:
    """Full result for one run: root causes, escalations and agent trajectories."""
    store = get_run_store()
    record = await asyncio.to_thread(store.get, run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"run {run_id} not found")
    log_event(logger, "run_retrieved", run_id=run_id, actor=principal.sub,
              source=record.summary.source)
    return record.to_dict()