"""Human-in-the-loop approval store.

Agents propose side-effecting actions; they land here as pending. A human with
the right role approves or rejects; only approved actions are executed. Starter
uses an in-memory store — back with the DB (notification_audit) in production.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class ApprovalItem:
    id: str
    run_id: str
    action_type: str            # send_notification | file_escalation
    payload: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decided_by: str | None = None
    decided_at: datetime | None = None


class ApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalItem] = {}

    def enqueue(self, run_id: str, action_type: str, payload: dict) -> ApprovalItem:
        item = ApprovalItem(id=uuid.uuid4().hex, run_id=run_id,
                            action_type=action_type, payload=payload)
        self._items[item.id] = item
        return item

    def list_pending(self, run_id: str | None = None) -> list[ApprovalItem]:
        return [
            i for i in self._items.values()
            if i.status == ApprovalStatus.PENDING and (run_id is None or i.run_id == run_id)
        ]

    def get(self, item_id: str) -> ApprovalItem | None:
        return self._items.get(item_id)

    def decide(self, item_id: str, approve: bool, actor: str) -> ApprovalItem:
        item = self._items[item_id]
        item.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        item.decided_by = actor
        item.decided_at = datetime.now(timezone.utc)
        return item

    def mark_executed(self, item_id: str) -> None:
        self._items[item_id].status = ApprovalStatus.EXECUTED


# Process-wide singleton for the starter.
_STORE = ApprovalStore()


def get_approval_store() -> ApprovalStore:
    return _STORE
