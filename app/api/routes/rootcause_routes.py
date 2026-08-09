"""Ranked top-N root-cause view.

Combines three tiers into one ranked list so callers can always request N causes:
  1. validated adverse findings   (cleared all five gates, worse than baseline)
  2. validated protective factors (cleared all gates, better than baseline)
  3. watchlist near-misses        (top rejected candidates, with failure reason)

Every row is tagged with its tier and evidence grade so nothing is presented as
a confirmed cause when it is not.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.analytics.ingest import ingest
from app.analytics.root_cause import GateParams, run_root_cause
from app.api.security import Principal, require_scope
from app.api.upload_validation import read_upload
from app.core import column_mapping as cm
from app.core.config import get_settings

logger = logging.getLogger("api.rootcause")
router = APIRouter(prefix="/v1/root-causes", tags=["root-cause"])


@router.post("")
async def top_root_causes(
    file: UploadFile = File(...),
    n: int = Query(default=5, ge=1, le=50),
    include_protective: bool = Query(default=True),
    include_watchlist: bool = Query(default=True),
    principal: Principal = Depends(require_scope("analytics:read")),
) -> dict:
    s = get_settings()
    df = await read_upload(file)
    ing = ingest(df)
    avg_margin = float(ing.closed[cm.BENEFIT_PER_ORDER].mean()) if cm.BENEFIT_PER_ORDER in ing.closed.columns else 0.0
    out = run_root_cause(
        ing.closed,
        GateParams(support_floor=s.support_floor, effect_size_min=s.effect_size_min,
                   fdr_q=s.fdr_q, confound_margin=s.confound_margin,
                   stability_var_max=s.stability_var_max),
        avg_margin=avg_margin,
    )

    ranked: list[dict] = []

    for rank, f in enumerate(out.findings, start=1):
        ranked.append({
            "rank": rank, "tier": "validated_root_cause",
            "label": f.label, "n": f.n,
            "late_rate": round(f.seg_rate, 4), "baseline_rate": round(f.baseline_rate, 4),
            "lift": round(f.lift, 3), "excess_orders": round(f.excess_orders, 1),
            "excess_margin_usd": round(f.excess_margin, 2),
            "confidence": round(f.confidence, 3),
            "evidence_grade": f.evidence_grade,
        })

    if include_protective:
        for f in out.protective:
            ranked.append({
                "rank": None, "tier": "protective_factor",
                "label": f.label, "n": f.n,
                "late_rate": round(f.seg_rate, 4), "baseline_rate": round(f.baseline_rate, 4),
                "lift": round(f.lift, 3), "excess_orders": round(f.excess_orders, 1),
                "confidence": round(f.confidence, 3),
                "evidence_grade": f.evidence_grade,
                "note": "performs better than baseline",
            })

    if include_watchlist and len(ranked) < n:
        near = [r for r in out.rejected
                if r.failed_gate in ("gate4_confound", "gate5_stability")
                and r.p_value is not None]
        near.sort(key=lambda r: r.p_value)
        for r in near:
            if len(ranked) >= n:
                break
            ranked.append({
                "rank": None, "tier": "watchlist",
                "label": r.pattern_id, "failed_gate": r.failed_gate,
                "reason": r.reason, "p_value": r.p_value,
                "evidence_grade": "hypothesis",
                "note": "did not clear validation; monitor only",
            })

    return {
        "run_id": out.__dict__.get("run_id", ""),
        "requested": n,
        "returned": len(ranked[:n]) if n else len(ranked),
        "global_late_rate": round(out.global_rate, 4),
        "candidates_enumerated": out.candidates_enumerated,
        "m_tests_conducted": out.m_tests_conducted,
        "validated_count": len(out.findings),
        "protective_count": len(out.protective),
        "top": ranked[:n],
    }
