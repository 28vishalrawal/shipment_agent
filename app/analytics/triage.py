"""Lane A deterministic scoring: risk, slip, impact. No LLM."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.core import column_mapping as cm
from app.analytics.rate_table import RateTable
from app.domain.models import EvidenceGrade, ReasonCode, TriageRecord

SEGMENT_WEIGHT = {"Corporate": 1.5, "Home Office": 1.2, "Consumer": 1.0}


def _reason_code(mode_rate: float, cell_lift: float) -> ReasonCode:
    if mode_rate > 0.95:
        return ReasonCode.STRUCTURAL_SLA
    if mode_rate >= 0.50:
        return ReasonCode.MODE_RISK
    if cell_lift > 1.15:
        return ReasonCode.CELL_RISK
    return ReasonCode.LOW_RISK


def score_open_orders(
    open_orders: pd.DataFrame,
    rt: RateTable,
    shrinkage_k: int,
    eta_percentile: int,
    queue_cap: int,
    today: date | None = None,
) -> list[TriageRecord]:
    if open_orders.empty:
        return []
    today = today or date.today()
    records: list[TriageRecord] = []

    for _, row in open_orders.iterrows():
        mode = str(row.get(cm.SHIPPING_MODE, ""))
        region = str(row.get(cm.ORDER_REGION, ""))
        category = str(row.get(cm.CATEGORY, ""))
        p_late, src = rt.shrunk_rate(mode, region, category, shrinkage_k)
        mode_rate = rt.mode_rate.get(mode, rt.global_rate)
        cell_lift = p_late / mode_rate if mode_rate else 1.0
        reason = _reason_code(mode_rate, cell_lift)

        eta_days = rt.eta_days(mode, eta_percentile)
        scheduled = rt.scheduled.get(mode, eta_days)
        slip = max(0, int(round(eta_days - scheduled)))
        revised_eta = today + timedelta(days=int(round(eta_days)))

        value = float(row.get(cm.ORDER_ITEM_TOTAL, row.get(cm.SALES, 0.0)) or 0.0)
        segment = str(row.get(cm.CUSTOMER_SEGMENT, "Consumer"))
        weight = SEGMENT_WEIGHT.get(segment, 1.0)
        impact = p_late * value * weight

        records.append(
            TriageRecord(
                order_id=str(row.get(cm.ORDER_ID, "")),
                p_late=round(p_late, 4),
                expected_slip_days=slip,
                revised_eta=revised_eta,
                reason_code=reason,
                value_at_risk=round(value, 2),
                segment=segment,
                confidence_source=src,
                impact_score=round(impact, 2),
                evidence_grade=EvidenceGrade.DATA_SUPPORTED,
            )
        )

    # Prioritise by impact. queue_cap <= 0 means "no cap": return ALL at-risk
    # orders (satisfies Goal 1's "draft a notification for each"). A positive cap
    # returns the top-N by impact plus a high-prob/high-value tail (alert-fatigue
    # control for ops who want a prioritised subset).
    records.sort(key=lambda r: r.impact_score, reverse=True)
    if queue_cap is None or queue_cap <= 0 or len(records) <= queue_cap:
        return records
    if records:
        vals = sorted((r.value_at_risk for r in records))
        p90 = vals[int(0.9 * (len(vals) - 1))]
    else:
        p90 = 0.0
    top = records[:queue_cap]
    tail = [r for r in records[queue_cap:] if r.p_late > 0.9 and r.value_at_risk >= p90]
    return top + tail