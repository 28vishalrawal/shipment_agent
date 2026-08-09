"""CLI entry: analyze a file end-to-end without the HTTP layer.

Usage: PYTHONPATH=. python scripts/run_cli.py data/synthetic_orders.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json

import pandas as pd

from app.agents.orchestrator import Orchestrator
from app.analytics.ingest import ingest
from app.core.config import get_settings
from app.providers.factory import build_provider


async def _run(path: str) -> None:
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    settings = get_settings()
    orch = Orchestrator(build_provider(settings), settings)
    triage, notifications, report = await orch.run(ingest(df))
    print(json.dumps({
        "run_id": report.run_id,
        "input_rows": report.input_rows,
        "global_late_rate": round(report.global_late_rate, 4),
        "candidates": report.candidates_enumerated,
        "m_tests": report.m_tests_conducted,
        "triaged": len(triage),
        "notifications": len(notifications),
        "findings": [
            {"label": f.label, "n": f.n, "rate": round(f.seg_rate, 3),
             "lift": round(f.lift, 2), "excess_orders": round(f.excess_orders),
             "confidence": round(f.confidence, 2), "evidence_grade": f.evidence_grade}
            for f in report.findings
        ],
        "rejected_count": len(report.rejected),
        "escalation": report.escalation.model_dump(),
    }, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    asyncio.run(_run(args.path))
