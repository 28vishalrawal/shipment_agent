"""Dataset resolution and RunContext caching for the MCP surface.

Why this layer exists
---------------------
Every tool in `app.tools.registry` expects a populated `RunContext` — the HTTP
path builds one per upload. An MCP client has no upload: a conversational agent
asks "what's driving lateness in the EU lane", it does not POST a CSV. So the
MCP surface resolves a *named dataset* from a directory the operator controls
and builds the RunContext itself.

Two constraints shape this module:

  * Path containment. The dataset name arrives from a language model, so it is
    untrusted input. Names are resolved against one allowed root and rejected if
    they escape it, regardless of how they are spelled.
  * Cost. `ingest()` parses and cleans the whole frame; `build_rate_table()`
    aggregates it. Re-running both on every tool call would make a three-tool
    agent turn intolerably slow, so contexts are cached and keyed on file mtime
    so an updated batch invalidates its own entry.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path

from app.agentic.context import RunContext
from app.analytics.ingest import ingest
from app.core.config import Settings
from app.observability.logging_setup import new_id
from app.triggers.automation import _read_batch

logger = logging.getLogger("mcp.datasets")

SUFFIXES = {".csv", ".xlsx", ".xls"}
LATEST = "latest"


class DatasetError(RuntimeError):
    """Raised when a dataset name cannot be resolved to an allowed file."""


def dataset_root(settings: Settings) -> Path:
    """The single directory MCP tools may read from.

    Defaults to the file-drop archive, which is where processed batches already
    land — so the conversational surface sees exactly what the pipeline has
    already run, and nothing else on the host.
    """
    raw = settings.mcp_dataset_dir or settings.trigger_archive_dir
    if not raw:
        raise DatasetError(
            "no dataset directory configured; set MCP_DATASET_DIR or TRIGGER_ARCHIVE_DIR"
        )
    return Path(raw).expanduser().resolve()


def list_datasets(settings: Settings, limit: int = 50) -> list[dict]:
    """Available batch files, newest first."""
    root = dataset_root(settings)
    if not root.is_dir():
        return []
    files = [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in SUFFIXES
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "modified": int(p.stat().st_mtime),
        }
        for p in files[:limit]
    ]


def resolve(settings: Settings, name: str) -> Path:
    """Map an untrusted dataset name onto a file inside the allowed root.

    Containment is checked after resolution rather than by scanning the input
    for "..", because the encodings of an escape attempt are open-ended while
    the set of allowed parents is exactly one.
    """
    root = dataset_root(settings)
    name = (name or "").strip()
    if not name:
        raise DatasetError("dataset name is required")

    if name.lower() == LATEST:
        available = list_datasets(settings, limit=1)
        if not available:
            raise DatasetError(f"no datasets found in {root}")
        name = available[0]["name"]

    # A bare filename is the only accepted shape. Anything carrying separators
    # is rejected outright so resolution never has to reason about traversal.
    if Path(name).name != name:
        raise DatasetError("dataset must be a plain file name, not a path")

    candidate = (root / name).resolve()
    if candidate.parent != root:
        raise DatasetError("dataset resolves outside the allowed directory")
    if candidate.suffix.lower() not in SUFFIXES:
        raise DatasetError(f"unsupported file type {candidate.suffix!r}")
    if not candidate.is_file():
        raise DatasetError(f"dataset {name!r} not found")
    return candidate


class ContextCache:
    """Bounded LRU of built RunContexts, keyed by (path, mtime).

    Contexts are mutable — tools write `ctx.triage`, `ctx.rate_table` and
    `ctx.scratch`. Sharing one across concurrent MCP calls is intentional and
    safe here because every exposed tool is read-only with respect to the
    business domain: the writes are memoised analysis, so a racing pair either
    recomputes identical values or reuses them. Action tools never reach this
    layer (see bridge.EXPOSURE_RULE).
    """

    def __init__(self, max_entries: int = 4) -> None:
        self._entries: OrderedDict[tuple[str, int], RunContext] = OrderedDict()
        self._max = max(1, max_entries)
        self._lock = threading.Lock()

    def get(self, settings: Settings, name: str) -> RunContext:
        path = resolve(settings, name)
        key = (str(path), int(path.stat().st_mtime))

        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit

        # Built outside the lock: ingest is slow, and holding a global lock
        # across it would serialise every unrelated dataset request.
        ctx = self._build(settings, path)

        with self._lock:
            self._entries[key] = ctx
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return ctx

    @staticmethod
    def _build(settings: Settings, path: Path) -> RunContext:
        df = _read_batch(path, settings)
        ingested = ingest(df)
        logger.info(
            "mcp_context_built file=%s rows=%d analysis_rows=%d",
            path.name, ingested.input_rows, ingested.analysis_rows,
        )
        run_id = new_id()
        return RunContext(run_id=run_id, correlation_id=run_id, ingested=ingested)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_cache: ContextCache | None = None
_cache_lock = threading.Lock()


def get_cache(settings: Settings) -> ContextCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = ContextCache(settings.mcp_context_cache_size)
    return _cache