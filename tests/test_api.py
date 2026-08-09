"""API integration tests: auth, RBAC, idempotency."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from scripts.make_synthetic_data import generate


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_session=None):
    import os
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["ENABLE_LLM_NOTIFICATIONS"] = "false"
    # config is cached; clear so env overrides apply
    from app.core.config import get_settings
    get_settings.cache_clear()
    d = tmp_path_factory.mktemp("data")
    csv = d / "orders.csv"
    generate(4000, seed=9).to_csv(csv, index=False)
    return TestClient(create_app()), str(csv)


def _token(c, role="analyst"):
    return c.post("/auth/token", json={"username": "u", "role": role}).json()["access_token"]


def test_health_ok(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_requires_auth(client):
    c, csv = client
    r = c.post("/v1/analyze", files={"file": ("o.csv", open(csv, "rb"), "text/csv")})
    assert r.status_code == 401


def test_viewer_forbidden(client):
    c, csv = client
    tok = _token(c, "viewer")
    r = c.post("/v1/analyze", files={"file": ("o.csv", open(csv, "rb"), "text/csv")},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


def test_analyst_can_run(client):
    c, csv = client
    tok = _token(c, "analyst")
    r = c.post("/v1/analyze", files={"file": ("o.csv", open(csv, "rb"), "text/csv")},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["m_tests_conducted"] > 0


def test_idempotent_replay(client):
    c, csv = client
    tok = _token(c, "analyst")
    h = {"Authorization": f"Bearer {tok}", "Idempotency-Key": "k-123"}
    r1 = c.post("/v1/analyze", files={"file": ("o.csv", open(csv, "rb"), "text/csv")}, headers=h)
    r2 = c.post("/v1/analyze", files={"file": ("o.csv", open(csv, "rb"), "text/csv")}, headers=h)
    assert r1.json()["run_id"] == r2.json()["run_id"]


def test_rejects_unsupported_filetype(client):
    c, _ = client
    tok = _token(c, "analyst")
    r = c.post("/v1/analyze", files={"file": ("o.txt", b"hello", "text/plain")},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 415
