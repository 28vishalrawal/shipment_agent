"""Per-run context object passed to every tool. Holds the loaded frames and the
accumulating results so tools compose without globals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app.analytics.ingest import IngestResult
from app.analytics.rate_table import RateTable
from app.domain.models import (
    NotificationRecord,
    RootCauseReport,
    TriageRecord,
)


@dataclass
class RunContext:
    run_id: str
    correlation_id: str
    ingested: Optional[IngestResult] = None
    rate_table: Optional[RateTable] = None
    triage: list[TriageRecord] = field(default_factory=list)
    notifications: list[NotificationRecord] = field(default_factory=list)
    report: Optional[RootCauseReport] = None
    # Approval queue: side-effecting actions the agent proposed but that must be
    # confirmed by a human before execution.
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
