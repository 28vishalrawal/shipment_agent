"""Automation triggers that launch agentic runs without a human initiating them.

Three entry points, one dispatch function:
  - schedule: periodic (cron-like) via asyncio loop.
  - file-drop: watch an inbox directory for new CSV/XLSX.
  - webhook: an authenticated POST that carries or references a batch.

All converge on dispatch_run(), so a run behaves identically regardless of trigger.
Idempotency: each source file is hashed; a hash seen before is skipped.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import pandas as pd

from app.agentic.orchestrator import AgenticOrchestrator
from app.analytics.ingest import ingest
from app.core.config import Settings
from app.observability.logging_setup import log_event, new_id
from app.providers.factory import build_provider

logger = logging.getLogger("triggers")

_SEEN_HASHES: set[str] = set()


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


async def dispatch_run(df: pd.DataFrame, settings: Settings, source: str) -> dict:
    correlation_id = new_id()
    log_event(logger, "trigger_fired", correlation_id=correlation_id, source=source,
              rows=len(df))
    ingested = ingest(df)
    orch = AgenticOrchestrator(build_provider(settings), settings)
    return await orch.run(ingested, correlation_id)


async def dispatch_file(path: Path, settings: Settings, source: str) -> dict | None:
    data = path.read_bytes()
    h = _hash_bytes(data)
    if h in _SEEN_HASHES:
        log_event(logger, "trigger_skipped_duplicate", source=source, file=path.name)
        return None
    _SEEN_HASHES.add(h)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    return await dispatch_run(df, settings, source)


# ---------------- scheduled trigger ----------------
async def run_scheduler(settings: Settings, inbox: Path, interval_s: int = 3600) -> None:
    """Periodically scan the inbox and process any new files. Cron in prod."""
    log_event(logger, "scheduler_started", interval_s=interval_s, inbox=str(inbox))
    while True:
        for p in sorted(inbox.glob("*.csv")) + sorted(inbox.glob("*.xlsx")):
            try:
                await dispatch_file(p, settings, source="scheduler")
            except Exception as exc:  # a bad file must not kill the loop
                log_event(logger, "trigger_error", status="error", source="scheduler",
                          file=p.name, error_code=type(exc).__name__)
        await asyncio.sleep(interval_s)


# ---------------- file-drop trigger ----------------
async def watch_inbox(settings: Settings, inbox: Path, poll_s: int = 5) -> None:
    """Poll-based file watcher (portable; swap for watchdog/inotify in prod)."""
    log_event(logger, "file_watch_started", inbox=str(inbox))
    inbox.mkdir(parents=True, exist_ok=True)
    processed: set[str] = set()
    while True:
        for p in sorted(inbox.glob("*.csv")) + sorted(inbox.glob("*.xlsx")):
            if p.name in processed:
                continue
            processed.add(p.name)
            try:
                await dispatch_file(p, settings, source="file_drop")
            except Exception as exc:
                log_event(logger, "trigger_error", status="error", source="file_drop",
                          file=p.name, error_code=type(exc).__name__)
        await asyncio.sleep(poll_s)
