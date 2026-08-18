"""Durable store for completed agentic runs.

Why this exists
---------------
Before this, a run's analysis existed only in the HTTP response body. That was
survivable while every run was a manual upload — the uploader was looking at the
response — but it breaks in two ways:

  * File-drop and scheduled runs have no HTTP caller at all. The watcher awaited
    dispatch_file() and discarded the result, so nobody could ever see the root
    causes for a dropped batch.
  * Approvals outlived the analysis. ApprovalStore persists escalations keyed by
    run_id, so a manager would see a $249K escalation in the queue with no way to
    reach the evidence behind it — the worst possible state for a
    human-in-the-loop system.

Runs therefore persist here, keyed by run_id, and every trigger writes the same
record. A manager can open any run regardless of who or what started it.

Backed by SQLite (stdlib sqlite3) rather than a dict, because the point of a
file-drop trigger is that batches arrive unattended — a run that lands overnight
must still be there in the morning, and a process restart must not erase it.
Writes go through a lock and callers wrap them in asyncio.to_thread, so the
event loop is never blocked by disk I/O.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("run.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    source          TEXT NOT NULL,
    triggered_by    TEXT,
    file_name       TEXT,
    rows            INTEGER,
    at_risk_orders  INTEGER,
    notifications   INTEGER,
    root_cause_count INTEGER,
    escalation_count INTEGER,
    status          TEXT NOT NULL,
    error           TEXT,
    result          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
"""


@dataclass
class RunSummary:
    """Row-level view for the list endpoint: everything needed to choose a run,
    without shipping the full result payload (which carries agent trajectories
    and can be large)."""

    run_id: str
    created_at: datetime
    source: str                      # upload | webhook | file_drop | scheduler
    triggered_by: str | None         # username, or the dropped file name
    file_name: str | None
    rows: int = 0
    at_risk_orders: int = 0
    notifications: int = 0
    root_cause_count: int = 0
    escalation_count: int = 0
    status: str = "completed"        # completed | failed
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class RunRecord:
    summary: RunSummary
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary.to_dict(), "result": self.result}


def _db_path(database_url: str) -> Path:
    """Filesystem path from a SQLAlchemy-style URL.

    DATABASE_URL is declared as sqlite+aiosqlite:///./data/app.db for the async
    ORM; this store uses stdlib sqlite3 against the same file so there is one
    database rather than two.
    """
    raw = database_url.split("///")[-1] if "///" in database_url else "./data/app.db"
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class RunStore:
    def __init__(self, database_url: str | None = None) -> None:
        settings = get_settings()
        self._path = _db_path(database_url or settings.database_url)
        # check_same_thread=False because asyncio.to_thread hands work to
        # arbitrary worker threads; _lock serialises access in exchange.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        source: str,
        triggered_by: str | None = None,
        file_name: str | None = None,
        rows: int = 0,
    ) -> RunSummary:
        summary = RunSummary(
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            source=source,
            triggered_by=triggered_by,
            file_name=file_name,
            rows=rows,
            at_risk_orders=int(result.get("at_risk_orders") or 0),
            notifications=int(result.get("notifications_drafted") or 0),
            root_cause_count=len(result.get("root_causes") or []),
            escalation_count=len(result.get("escalations") or []),
            status="completed",
        )
        self._insert(summary, result)
        logger.info("run_persisted run_id=%s source=%s", run_id, source)
        return summary

    def save_failure(
        self, run_id: str, error: str, *, source: str,
        triggered_by: str | None = None, file_name: str | None = None,
    ) -> RunSummary:
        """Record a run that never produced a result.

        A failed drop that leaves no trace is indistinguishable from a file that
        was never picked up — the operator needs to see that it was attempted.
        """
        summary = RunSummary(
            run_id=run_id, created_at=datetime.now(timezone.utc), source=source,
            triggered_by=triggered_by, file_name=file_name,
            status="failed", error=error[:500],
        )
        self._insert(summary, {})
        return summary

    def _insert(self, s: RunSummary, result: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO runs (run_id, created_at, source, triggered_by,
                   file_name, rows, at_risk_orders, notifications, root_cause_count,
                   escalation_count, status, error, result)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (s.run_id, s.created_at.isoformat(), s.source, s.triggered_by,
                 s.file_name, s.rows, s.at_risk_orders, s.notifications,
                 s.root_cause_count, s.escalation_count, s.status, s.error,
                 json.dumps(result, default=str)),
            )
            self._conn.commit()

    def list(self, limit: int = 50, source: str | None = None) -> list[RunSummary]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            params.append(source)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._to_summary(r) for r in rows]

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            result = json.loads(row["result"])
        except (ValueError, TypeError):
            result = {}
        return RunRecord(summary=self._to_summary(row), result=result)

    def latest(self) -> RunRecord | None:
        runs = self.list(limit=1)
        return self.get(runs[0].run_id) if runs else None

    @staticmethod
    def _to_summary(row: sqlite3.Row) -> RunSummary:
        return RunSummary(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            source=row["source"], triggered_by=row["triggered_by"],
            file_name=row["file_name"], rows=row["rows"] or 0,
            at_risk_orders=row["at_risk_orders"] or 0,
            notifications=row["notifications"] or 0,
            root_cause_count=row["root_cause_count"] or 0,
            escalation_count=row["escalation_count"] or 0,
            status=row["status"], error=row["error"],
        )


_store: RunStore | None = None
_store_lock = threading.Lock()


def get_run_store() -> RunStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RunStore()
    return _store


def reset_run_store() -> None:
    """Drop the singleton so the next call reopens against the current
    DATABASE_URL. Used by tests; not part of the runtime path."""
    global _store
    with _store_lock:
        if _store is not None:
            _store._conn.close()
        _store = None