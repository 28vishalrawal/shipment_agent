"""Domain models. Pydantic everywhere so every object is schema-validated.

Central design rule: every quantitative claim carries an EvidenceGrade so the
system can never blur an observed fact into a hypothesis.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceGrade(str, Enum):
    OBSERVED_FACT = "observed_fact"          # directly calculated from order data
    DATA_SUPPORTED = "data_supported_risk"   # statistically supported relationship
    HYPOTHESIS = "hypothesis"                # possible cause; needs ops validation


class ReasonCode(str, Enum):
    STRUCTURAL_SLA = "structural_sla"   # mode late rate > 0.95
    MODE_RISK = "mode_risk"             # 0.5-0.95
    CELL_RISK = "cell_risk"             # cell lift > 1.15
    LOW_RISK = "low_risk"


class RemedyTier(int, Enum):
    APOLOGY_ONLY = 1
    EXPEDITE_OFFER = 2
    PARTIAL_SHIP_CONTACT = 3


class DeliveryOutcome(str, Enum):
    LATE = "Late delivery"
    ADVANCE = "Advance shipping"
    ON_TIME = "Shipping on time"
    CANCELED = "Shipping canceled"


class TriageRecord(BaseModel):
    order_id: str
    p_late: float = Field(ge=0.0, le=1.0)
    expected_slip_days: int
    revised_eta: date
    reason_code: ReasonCode
    value_at_risk: float
    segment: str
    confidence_source: str  # cell | mode | global
    impact_score: float
    evidence_grade: EvidenceGrade = EvidenceGrade.DATA_SUPPORTED


class NotificationRecord(BaseModel):
    order_id: str
    channel: str = "email"
    subject: str
    body: str
    remedy_tier: RemedyTier
    grounded_fields: list[str]
    validator_pass: bool
    used_fallback: bool = False


class CandidatePattern(BaseModel):
    pattern_id: str
    dims: dict[str, str]
    n: int
    seg_rate: float
    baseline_key: str
    baseline_rate: float
    lift: float
    excess_orders: float
    excess_margin: float
    p_value: float
    p_adj: Optional[float] = None


class GateResult(BaseModel):
    gate: str
    passed: bool
    reason: str


class ValidatedFinding(BaseModel):
    pattern_id: str
    label: str
    dims: dict[str, str]
    n: int
    seg_rate: float
    baseline_rate: float
    lift: float
    excess_orders: float
    excess_margin: float
    p_value: float
    confidence: float
    gates: list[GateResult]
    evidence_grade: EvidenceGrade
    narrative: Optional[str] = None
    mitigation: Optional[str] = None
    expected_effect: Optional[str] = None


class RejectedCandidate(BaseModel):
    pattern_id: str
    dims: dict[str, str]
    failed_gate: str
    reason: str
    p_value: Optional[float] = None


class EscalationDecision(BaseModel):
    run_id: str
    candidates_evaluated: int
    m_tests_conducted: int
    escalated: bool
    # Position of this finding in the ranked root-cause list (1 = strongest).
    # A run may raise several escalations; rank orders them for the operator.
    rank: int = 1
    finding_id: Optional[str] = None
    confidence: float
    threshold: float
    suppression_reason: Optional[str] = None
    # Inline explanation of the escalated finding so the escalation is
    # self-contained (no need to cross-reference the findings array).
    finding_label: Optional[str] = None
    excess_orders: Optional[float] = None
    excess_margin_usd: Optional[float] = None
    narrative: Optional[str] = None
    mitigation: Optional[str] = None
    expected_effect: Optional[str] = None


class RootCauseReport(BaseModel):
    run_id: str
    generated_at: datetime
    input_rows: int
    analysis_rows: int
    global_late_rate: float
    candidates_enumerated: int
    m_tests_conducted: int
    findings: list[ValidatedFinding]
    rejected: list[RejectedCandidate]
    escalation: EscalationDecision