"""Automation triggers that launch agentic runs without a human initiating them.

Three entry points, one dispatch function:
  - schedule: periodic (cron-like) via asyncio loop.
  - file-drop: watch an inbox directory for new CSV/XLSX.
  - webhook: an authenticated POST that carries or references a batch.

All converge on dispatch_run(), so a run behaves identically regardless of trigger.

File lifecycle (file-drop and scheduler):
    <inbox>/batch.csv
        -> processed -> <archive>/20260817T120000Z__batch.csv
        -> failed    -> <archive>/failed/20260817T120000Z__batch.csv

Moving the file out of the inbox *is* the completion record. That makes the
watcher restart-safe without a database: anything still in the inbox has not
been processed, so an in-memory "seen" set cannot silently lose that fact
across a restart.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.agentic.orchestrator import AgenticOrchestrator
from app.analytics.ingest import ingest
from app.core.config import Settings
from app.observability.logging_setup import log_event, new_id
from app.persistence.run_store import get_run_store
from app.providers.factory import build_provider

logger = logging.getLogger("triggers")

SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls")

_SEEN_HASHES: set[str] = set()


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iter_batches(inbox: Path) -> list[Path]:
    """Files in the inbox worth considering, oldest first.

    Sorted by modification time so a backlog is processed in arrival order.
    Dotfiles are skipped because many copy tools stage into `.name.part` first.
    """
    found = [
        p for p in inbox.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(".")
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def _archive(path: Path, archive_dir: Path, failed: bool = False) -> Path | None:
    """Move a finished file out of the inbox, timestamped to avoid collisions.

    Never raises: failing to archive must not take down the watcher, but it is
    logged loudly because the file would otherwise be reprocessed on the next
    poll and loop indefinitely.
    """
    target_dir = archive_dir / "failed" if failed else archive_dir
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{_timestamp()}__{path.name}"
        n = 1
        while target.exists():
            target = target_dir / f"{_timestamp()}__{n}__{path.name}"
            n += 1
        # shutil.move rather than Path.rename: the archive may sit on a different
        # filesystem or volume mount, which rename cannot cross.
        shutil.move(str(path), str(target))
        return target
    except Exception as exc:
        log_event(logger, "trigger_archive_failed", status="error",
                  file=path.name, error_code=type(exc).__name__,
                  error_detail=str(exc)[:200])
        return None


async def dispatch_run(
    df: pd.DataFrame,
    settings: Settings,
    source: str,
    *,
    triggered_by: str | None = None,
    file_name: str | None = None,
) -> dict:
    """Run the agentic pipeline and persist the result.

    Saving here rather than at each call site means every trigger — upload,
    webhook, file drop, scheduler — produces an identical, retrievable record.
    Previously a file-drop run's analysis was discarded the moment this returned,
    so nobody could see the root causes behind a dropped batch.
    """
    correlation_id = new_id()
    log_event(logger, "trigger_fired", correlation_id=correlation_id, source=source,
              rows=len(df))
    ingested = ingest(df)
    orch = AgenticOrchestrator(build_provider(settings), settings)
    store = get_run_store()
    try:
        result = await orch.run(ingested, correlation_id)
    except Exception as exc:
        # A failed drop that leaves no trace is indistinguishable from a file
        # that was never picked up, so record the attempt before re-raising.
        await asyncio.to_thread(
            store.save_failure, correlation_id, f"{type(exc).__name__}: {exc}",
            source=source, triggered_by=triggered_by, file_name=file_name,
        )
        raise
    await asyncio.to_thread(
        store.save, result.get("run_id") or correlation_id, result,
        source=source, triggered_by=triggered_by, file_name=file_name, rows=len(df),
    )
    return result


def _read_batch(path: Path, settings: Settings) -> pd.DataFrame:
    """Parse a dropped file, applying the same limits as the HTTP upload path.

    Blocking (pandas), so callers run it in a worker thread.
    """
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise ValueError(
            f"file is {size_mb:.1f}MB, over the {settings.max_upload_mb}MB limit"
        )
    if path.suffix.lower() == ".csv":
        # Mirrors read_upload()'s encoding ladder: this dataset family ships as
        # latin-1, which fails a strict utf-8 read.
        for encoding in ("utf-8", "ISO-8859-1", "hp-roman8"):
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError("could not decode CSV as utf-8, ISO-8859-1 or hp-roman8")
    return pd.read_excel(path)


async def dispatch_file(
    path: Path, settings: Settings, source: str, archive_dir: Path | None = None
) -> dict | None:
    """Process one dropped file, then move it to the archive.

    Returns the run result, or None when the file was skipped as a duplicate.
    The file is archived on success *and* on failure, so a file that cannot be
    parsed does not sit in the inbox failing on every poll.
    """
    failed = False
    try:
        data = await asyncio.to_thread(path.read_bytes)
        h = _hash_bytes(data)
        if h in _SEEN_HASHES:
            log_event(logger, "trigger_skipped_duplicate", source=source,
                      file=path.name, content_hash=h)
            return None
        _SEEN_HASHES.add(h)
        # Parsing and the run itself are CPU-heavy; keep them off the event loop
        # so the API stays responsive while a batch is processing.
        df = await asyncio.to_thread(_read_batch, path, settings)
        return await dispatch_run(df, settings, source,
                                  triggered_by=path.name, file_name=path.name)
    except Exception:
        failed = True
        raise
    finally:
        if archive_dir is not None:
            moved = _archive(path, archive_dir, failed=failed)
            if moved:
                log_event(logger, "trigger_file_archived", source=source,
                          file=path.name, archived_to=str(moved), failed=failed)


async def _process_once(
    inbox: Path, archive: Path, settings: Settings, source: str, stable: dict[str, int]
) -> None:
    """One sweep of the inbox. Never raises: the caller's loop must survive."""
    try:
        entries = await asyncio.to_thread(_iter_batches, inbox)
    except FileNotFoundError:
        inbox.mkdir(parents=True, exist_ok=True)
        return

    live: set[str] = set()
    for p in entries:
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            continue  # moved or deleted between listing and stat
        live.add(p.name)

        # Require the size to hold steady across consecutive polls before
        # reading: a file still being copied in would otherwise be parsed
        # half-written, and might even succeed on truncated data.
        key = p.name
        if stable.get(key) != size:
            stable[key] = size
            stable[f"{key}::count"] = 1
            continue
        count = stable.get(f"{key}::count", 1) + 1
        stable[f"{key}::count"] = count
        if count < max(2, settings.trigger_stable_polls):
            continue
        if size == 0:
            continue  # empty placeholder; wait for content

        try:
            await dispatch_file(p, settings, source=source, archive_dir=archive)
        except Exception as exc:
            log_event(logger, "trigger_error", status="error", source=source,
                      file=p.name, error_code=type(exc).__name__,
                      error_detail=str(exc)[:300])
        finally:
            stable.pop(key, None)
            stable.pop(f"{key}::count", None)

    # Drop tracking for files that have left the inbox.
    for k in [k for k in stable if k.split("::")[0] not in live]:
        stable.pop(k, None)


# ---------------- scheduled trigger ----------------
async def run_scheduler(
    settings: Settings, inbox: Path, archive: Path | None = None, interval_s: int = 3600
) -> None:
    """Periodically scan the inbox and process any new files. Cron in prod."""
    archive = archive or inbox.parent / "archive"
    log_event(logger, "scheduler_started", interval_s=interval_s, inbox=str(inbox))
    stable: dict[str, int] = {}
    while True:
        await _process_once(inbox, archive, settings, "scheduler", stable)
        await asyncio.sleep(interval_s)


# ---------------- file-drop trigger ----------------
async def watch_inbox(
    settings: Settings,
    inbox: Path,
    archive: Path | None = None,
    poll_s: int | None = None,
) -> None:
    """Poll-based file watcher (portable; swap for watchdog/inotify in prod).

    Runs until cancelled. Processed files are moved to `archive`, which is what
    makes this restart-safe: the inbox always holds exactly the work outstanding.
    """
    archive = archive or inbox.parent / "archive"
    poll = poll_s or settings.trigger_poll_s
    inbox.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    log_event(logger, "file_watch_started", inbox=str(inbox), archive=str(archive),
              poll_s=poll)
    stable: dict[str, int] = {}
    try:
        while True:
            await _process_once(inbox, archive, settings, "file_drop", stable)
            await asyncio.sleep(poll)
    except asyncio.CancelledError:
        log_event(logger, "file_watch_stopped", inbox=str(inbox))
        raise