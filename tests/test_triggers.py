"""File-drop trigger: processing, archiving, quarantine and restart safety."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.triggers import automation
from app.triggers.automation import _process_once, watch_inbox


def _frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"order_id": range(n), "value": range(n)})


def _names(p: Path) -> list[str]:
    return sorted(x.name for x in p.iterdir()) if p.exists() else []


@pytest.fixture(autouse=True)
def _reset_seen():
    automation._SEEN_HASHES.clear()
    yield
    automation._SEEN_HASHES.clear()


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    inbox, archive = tmp_path / "input", tmp_path / "archive"
    inbox.mkdir()
    return inbox, archive


@pytest.fixture
def captured(monkeypatch) -> list[tuple[int, str]]:
    """Replace the orchestrator run so tests exercise file mechanics only."""
    seen: list[tuple[int, str]] = []

    async def fake(df, settings, source, **kwargs):
        seen.append((len(df), source))
        return {"rows": len(df)}

    monkeypatch.setattr(automation, "dispatch_run", fake)
    return seen


async def _sweep(inbox: Path, archive: Path, settings: Settings, times: int = 2) -> None:
    """Run enough polls to clear the stability threshold."""
    stable: dict[str, int] = {}
    for _ in range(times):
        await _process_once(inbox, archive, settings, "file_drop", stable)


@pytest.mark.asyncio
async def test_dropped_file_is_processed_then_archived(dirs, captured):
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2)
    _frame().to_csv(inbox / "batch.csv", index=False)

    await _sweep(inbox, archive, s)

    assert captured == [(20, "file_drop")]
    assert _names(inbox) == [], "processed file must leave the inbox"
    assert any(n.endswith("__batch.csv") for n in _names(archive))


@pytest.mark.asyncio
async def test_file_is_not_read_until_size_is_stable(dirs, captured):
    """A copy still in progress must not be parsed half-written."""
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2)
    target = inbox / "growing.csv"
    target.write_bytes(b"order_id,value\n1,1\n")

    stable: dict[str, int] = {}
    await _process_once(inbox, archive, s, "file_drop", stable)
    assert captured == [], "read on first sight, before size settled"

    target.write_bytes(b"order_id,value\n1,1\n2,2\n")   # still growing
    await _process_once(inbox, archive, s, "file_drop", stable)
    assert captured == [], "read while still changing"

    await _process_once(inbox, archive, s, "file_drop", stable)
    assert len(captured) == 1, "never processed once stable"
    assert _names(inbox) == []


@pytest.mark.asyncio
async def test_rejected_file_is_quarantined_not_retried(dirs, captured):
    """An unusable file must leave the inbox, or it fails on every poll forever."""
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2, max_upload_mb=0)  # rejects any file
    _frame().to_csv(inbox / "toobig.csv", index=False)

    await _sweep(inbox, archive, s)

    assert captured == [], "run should not have started"
    assert _names(inbox) == [], "bad file left in inbox would retry forever"
    assert any("toobig" in n for n in _names(archive / "failed"))


@pytest.mark.asyncio
async def test_duplicate_content_is_skipped_but_still_archived(dirs, captured):
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2)
    df = _frame()
    df.to_csv(inbox / "first.csv", index=False)
    await _sweep(inbox, archive, s)
    assert len(captured) == 1

    df.to_csv(inbox / "second.csv", index=False)   # identical bytes, new name
    await _sweep(inbox, archive, s)

    assert len(captured) == 1, "same content processed twice"
    assert _names(inbox) == [], "duplicate must still be cleared from the inbox"


@pytest.mark.asyncio
async def test_unsupported_extensions_are_left_alone(dirs, captured):
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2)
    (inbox / "notes.txt").write_text("not a batch")
    (inbox / ".partial.csv").write_text("a,b\n1,2\n")   # staged copy

    await _sweep(inbox, archive, s)

    assert captured == []
    assert _names(inbox) == [".partial.csv", "notes.txt"]


@pytest.mark.asyncio
async def test_archive_name_collision_does_not_overwrite(dirs, captured):
    inbox, archive = dirs
    s = Settings(trigger_stable_polls=2)
    for i in range(2):
        pd.DataFrame({"order_id": [i], "value": [i]}).to_csv(inbox / "same.csv", index=False)
        await _sweep(inbox, archive, s)

    archived = [n for n in _names(archive) if n.endswith("same.csv")]
    assert len(archived) == 2, f"an archived file was overwritten: {archived}"


@pytest.mark.asyncio
async def test_watcher_creates_dirs_and_stops_cleanly(tmp_path, captured):
    inbox, archive = tmp_path / "input", tmp_path / "archive"
    s = Settings(trigger_poll_s=1)
    task = asyncio.create_task(watch_inbox(s, inbox, archive, poll_s=1))
    await asyncio.sleep(0.3)

    assert inbox.exists() and archive.exists(), "watcher must create its directories"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task