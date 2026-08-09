"""Integration test: full orchestrator run with the mock provider (no network)."""
from __future__ import annotations

import pytest

from app.agents.orchestrator import Orchestrator
from app.analytics.ingest import ingest
from app.core.config import Settings
from app.providers.factory import build_provider
from scripts.make_synthetic_data import generate


@pytest.mark.asyncio
async def test_full_run_produces_report_and_suppresses_or_escalates():
    settings = Settings(llm_provider="mock", enable_llm_notifications=False, environment="local")
    provider = build_provider(settings)
    orch = Orchestrator(provider, settings)

    df = generate(6000, seed=5)
    ingested = ingest(df)
    triage, notifications, report = await orch.run(ingested)

    # Structural findings should exist (First Class 100% late by construction).
    assert report.m_tests_conducted > 0
    assert report.candidates_enumerated >= report.m_tests_conducted

    # Escalation decision is always present and internally consistent.
    d = report.escalation
    assert d.escalated == (d.confidence >= d.threshold)
    if not d.escalated:
        assert d.suppression_reason

    # Every notification is schema-valid and passed its validator (or fell back).
    for n in notifications:
        assert n.validator_pass


@pytest.mark.asyncio
async def test_no_findings_yields_silent_gate():
    # Uniform random data with no structure -> no systemic finding -> suppression.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({
        "Order Id": np.arange(n),
        "Delivery Status": rng.choice(["Late delivery", "Shipping on time"], n),
        "Late_delivery_risk": rng.integers(0, 2, n),
        "Shipping Mode": rng.choice(["A", "B"], n),
        "Order Region": rng.choice(["R1", "R2"], n),
        "Order Status": "COMPLETE",
    })
    settings = Settings(llm_provider="mock", enable_llm_notifications=False)
    orch = Orchestrator(build_provider(settings), settings)
    _, _, report = await orch.run(ingest(df))
    assert not report.escalation.escalated
    assert report.escalation.suppression_reason
