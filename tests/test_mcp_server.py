"""Tests for the MCP surface.

Weighted toward the boundary properties rather than tool output, because the
analytics behind these tools is already covered by test_analytics.py. What is
new here is a second, differently-authenticated door into the same functions —
so these tests assert what that door will not open.
"""
from __future__ import annotations

import asyncio

import pytest

from scripts.make_synthetic_data import generate

from app.core.config import Settings
from app.mcp_server import datasets
from app.mcp_server.bridge import exposed_tools, register_registry_tools
from app.tools.base import Tool, ToolRegistry
from app.tools.registry import build_registry


@pytest.fixture
def dataset_dir(tmp_path):
    d = tmp_path / "archive"
    d.mkdir()
    # The real generator, so the frame satisfies ingest()'s required columns.
    generate(400, seed=11).to_csv(d / "batch.csv", index=False)
    return d


@pytest.fixture
def settings(dataset_dir):
    return Settings(mcp_enabled=True, mcp_bearer_token="tok",
                    mcp_dataset_dir=str(dataset_dir))


# --------------------------------------------------------------- exposure rule
def test_approval_tools_are_never_exposed():
    names = {t.name for t in exposed_tools(build_registry())}
    assert "propose_customer_notification" not in names
    assert "propose_ops_escalation" not in names
    assert "run_root_cause_analysis" in names


def test_exposure_rule_is_structural_not_an_allowlist():
    """A newly added action tool must be excluded without touching this module."""
    from pydantic import BaseModel

    class Args(BaseModel):
        pass

    reg = build_registry()
    reg.register(
        Tool("delete_everything", "d", Args, lambda a, c: None, requires_approval=True)
    )
    assert "delete_everything" not in {t.name for t in exposed_tools(reg)}


def test_registration_publishes_only_safe_tools(settings):
    class FakeServer:
        def __init__(self):
            self.added = []

        def add_tool(self, fn, name=None, description=None):
            self.added.append(name)

    server = FakeServer()
    published = register_registry_tools(server, build_registry(), settings)
    assert server.added == published
    assert not any("propose" in n for n in published)


# ----------------------------------------------------------- path containment
@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "subdir/batch.csv",
        "",
    ],
)
def test_traversal_attempts_rejected(settings, name):
    with pytest.raises(datasets.DatasetError):
        datasets.resolve(settings, name)


def test_symlink_out_of_root_rejected(settings, dataset_dir, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a,b\n1,2\n")
    (dataset_dir / "link.csv").symlink_to(secret)
    # resolve() follows the link, so containment is judged on the real target.
    with pytest.raises(datasets.DatasetError):
        datasets.resolve(settings, "link.csv")


def test_unsupported_suffix_rejected(settings, dataset_dir):
    (dataset_dir / "notes.txt").write_text("hi")
    with pytest.raises(datasets.DatasetError):
        datasets.resolve(settings, "notes.txt")


def test_latest_resolves_and_lists(settings):
    assert datasets.resolve(settings, "latest").name == "batch.csv"
    assert [d["name"] for d in datasets.list_datasets(settings)] == ["batch.csv"]


def test_cache_invalidates_on_mtime_change(settings, dataset_dir):
    cache = datasets.ContextCache(max_entries=2)
    first = cache.get(settings, "batch.csv")
    assert cache.get(settings, "batch.csv") is first

    import os
    import time

    time.sleep(0.01)
    path = dataset_dir / "batch.csv"
    os.utime(path, (time.time() + 10, time.time() + 10))
    assert cache.get(settings, "batch.csv") is not first


# ------------------------------------------------------------ schema fidelity
def test_bridged_schema_keeps_constraints_and_dataset_arg(settings):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="t")
    register_registry_tools(server, build_registry(), settings)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    schema = tools["score_triage_queue"].input_schema
    assert "dataset" in schema["properties"]
    assert "dataset" in schema["required"]
    # ge=1, le=200 on TopNArgs.n must survive the projection, or the schema
    # would advertise values the tool rejects at validation time.
    assert schema["properties"]["n"]["minimum"] == 1
    assert schema["properties"]["n"]["maximum"] == 200


def test_unknown_dimension_returns_error_not_exception(settings):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="t")
    register_registry_tools(server, build_registry(), settings)
    result = asyncio.run(
        server.call_tool(
            "segment_late_rate",
            {"dataset": "batch.csv", "dimension": "no_such_column", "value": "x"},
        )
    )
    assert "error" in str(result).lower()


# ---------------------------------------------------------------------- auth
def _asgi_scope(headers):
    return {"type": "http", "path": "/mcp/",
            "headers": [(k.encode(), v.encode()) for k, v in headers]}


def _run_auth(headers):
    from app.mcp_server.auth import BearerAuthMiddleware

    seen = {"called": False, "status": None}

    async def downstream(scope, receive, send):
        seen["called"] = True

    async def send(msg):
        if msg["type"] == "http.response.start":
            seen["status"] = msg["status"]

    mw = BearerAuthMiddleware(downstream, "correct-token")
    asyncio.run(mw(_asgi_scope(headers), None, send))
    return seen


def test_auth_accepts_correct_token():
    assert _run_auth([("authorization", "Bearer correct-token")])["called"]


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("authorization", "Bearer wrong-token")],
        [("authorization", "correct-token")],       # missing scheme
        [("authorization", "Basic correct-token")],
        [("authorization", "Bearer ")],
        [("authorization", "Bearer correct-token-extra")],
    ],
)
def test_auth_rejects_everything_else(headers):
    seen = _run_auth(headers)
    assert not seen["called"]
    assert seen["status"] == 401


def test_empty_token_is_refused_at_construction():
    from app.mcp_server.auth import BearerAuthMiddleware

    with pytest.raises(ValueError):
        BearerAuthMiddleware(lambda *a: None, "")


def test_malformed_dataset_reported_not_raised(settings, dataset_dir):
    """A CSV that parses but is not an orders export must be a correctable
    message. Raising here would hand a sandboxed agent an opaque failure with
    no indication that choosing another file would work."""
    from mcp.server.mcpserver import MCPServer

    (dataset_dir / "wrong.csv").write_text("a,b\n1,2\n3,4\n")
    server = MCPServer(name="t")
    register_registry_tools(server, build_registry(), settings)
    result = asyncio.run(server.call_tool("summarize_data", {"dataset": "wrong.csv"}))
    text = str(result).lower()
    assert "error" in text and "required columns" in text
