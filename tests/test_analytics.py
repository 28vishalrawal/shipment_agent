"""Deterministic analytics tests: classification, scoring, gates, suppression."""
from __future__ import annotations

import pandas as pd
import pytest

from app.analytics.ingest import ingest
from app.analytics.rate_table import build_rate_table
from app.analytics.root_cause import GateParams, run_root_cause
from app.analytics.triage import score_open_orders
from app.core import column_mapping as cm
from scripts.make_synthetic_data import generate


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return generate(6000, seed=1)


def test_late_label_matches_delivery_status(data):
    res = ingest(data)
    late_from_status = (res.closed[cm.DELIVERY_STATUS] == "Late delivery").sum()
    assert res.closed["late"].sum() == late_from_status


def test_cancelled_excluded_from_analysis(data):
    res = ingest(data)
    assert (res.closed[cm.DELIVERY_STATUS] == "Shipping canceled").sum() == 0


def test_first_class_is_structurally_late(data):
    res = ingest(data)
    rt = build_rate_table(res.closed)
    # First Class scheduled 1, actual always 2 => 100% late by construction.
    assert rt.mode_rate["First Class"] == pytest.approx(1.0, abs=1e-9)


def test_small_samples_do_not_escalate():
    # A tiny anomalous segment must never survive to a finding.
    df = generate(400, seed=3)
    # Force a rare 3-way cell to be extreme but tiny.
    res = ingest(df)
    out = run_root_cause(res.closed, GateParams(support_floor=200))
    for f in out.findings:
        assert f.n >= 200  # nothing below the floor can be a finding


def test_effect_size_gate_filters_trivial(data):
    res = ingest(data)
    out = run_root_cause(res.closed, GateParams(effect_size_min=0.15))
    for f in out.findings:
        assert abs(f.lift - 1) >= 0.15


def test_m_tests_is_recorded_and_positive(data):
    res = ingest(data)
    out = run_root_cause(res.closed, GateParams())
    assert out.m_tests_conducted > 0
    assert out.candidates_enumerated >= out.m_tests_conducted


def test_priority_scoring_orders_by_impact(data):
    res = ingest(data)
    rt = build_rate_table(res.closed)
    recs = score_open_orders(res.open_orders, rt, shrinkage_k=50, eta_percentile=75,
                             queue_cap=1000)
    scores = [r.impact_score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_rejected_register_is_populated(data):
    res = ingest(data)
    out = run_root_cause(res.closed, GateParams())
    # Many candidates should be rejected (support floor alone removes a lot).
    assert len(out.rejected) > 0
    assert all(r.failed_gate for r in out.rejected)
