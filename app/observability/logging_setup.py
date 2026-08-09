"""Structured JSON logging with a strict allowlist of fields and a PII denylist."""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings

# Keys that must never be logged even if a caller passes them.
_DENY = {"api_key", "authorization", "email", "phone", "address", "prompt_raw", "pii"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        s = get_settings()
        base: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": s.environment,
            "service_name": s.service_name,
            "version": s.build_version,
            "level": record.levelname,
            "event_name": getattr(record, "event_name", record.getMessage()),
        }
        for k, v in getattr(record, "extra_fields", {}).items():
            if k.lower() in _DENY:
                continue
            base[k] = v
        return json.dumps(base, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def log_event(logger: logging.Logger, event_name: str, status: str = "ok", **fields: Any) -> None:
    safe = {k: v for k, v in fields.items() if k.lower() not in _DENY}
    logger.info(event_name, extra={"event_name": event_name, "extra_fields": {"event_status": status, **safe}})


def new_id() -> str:
    return uuid.uuid4().hex
