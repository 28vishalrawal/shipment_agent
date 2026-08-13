"""Output guardrails: block unsupported promises and ungrounded claims."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# Phrases we never allow a customer message to assert unless grounded/authorised.
PROHIBITED = [
    (re.compile(r"\brefund\b", re.I), "refund_promise"),
    (re.compile(r"\bcompensat", re.I), "compensation_promise"),
    (re.compile(r"\bcoupon|voucher|store credit\b", re.I), "compensation_promise"),
    (re.compile(r"\bguarantee\b", re.I), "guarantee_claim"),
    (re.compile(r"\bcurrently (?:in|at|near)\b", re.I), "tracking_location_claim"),
    (re.compile(r"\bunforeseen circumstances\b", re.I), "boilerplate_phrase"),
    (re.compile(r"\bwarehouse (?:fire|strike|accident)\b", re.I), "unsupported_internal_cause"),
]

DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2})",
    re.I,
)


@dataclass
class OutputCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_notification(
    body: str,
    subject: str,
    revised_eta: date,
    grounded_fields: dict,
    require_date: bool = True,
    min_grounded: int = 3,
    max_words: int = 180,
) -> OutputCheck:
    reasons: list[str] = []

    # A blank subject or body is never acceptable — force the deterministic
    # fallback (which always has both). Without this, an LLM that returns an
    # empty subject would pass and, via template pooling, propagate the empty
    # subject to every order sharing that message shape.
    if not subject or not subject.strip():
        reasons.append("empty_subject")
    if not body or not body.strip():
        reasons.append("empty_body")

    text = f"{subject}\n{body}"

    for pat, label in PROHIBITED:
        if pat.search(text):
            reasons.append(f"prohibited:{label}")

    if len(body.split()) > max_words:
        reasons.append("too_long")

    if require_date and not DATE_RE.search(text):
        reasons.append("no_date_present")

    # Every date mentioned must not precede the supported revised ETA.
    for m in DATE_RE.finditer(text):
        # Only structured ISO dates are strictly comparable; skip prose forms.
        iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}", m.group(0))
        if iso:
            try:
                if date.fromisoformat(m.group(0)) < revised_eta:
                    reasons.append("eta_earlier_than_supported")
            except ValueError:
                pass

    # Grounding: require min_grounded field values verbatim, but never demand
    # more than the number of non-empty fields actually supplied.
    available = [v for v in grounded_fields.values() if v and str(v).strip()]
    required = min(min_grounded, len(available))
    present = sum(1 for val in available if str(val) in text)
    if present < required:
        reasons.append(f"insufficient_grounding:{present}<{required}")

    return OutputCheck(ok=len(reasons) == 0, reasons=reasons)


def fallback_notification(
    order_id: str, product: str, revised_eta: date, quantity: str = ""
) -> tuple[str, str]:
    """Deterministic, always-safe message used when LLM output fails validation."""
    subject = f"Update on your order {order_id}"
    body = (
        f"Hello,\n\nWe want to let you know that your order {order_id} "
        f"containing {product} is now expected to arrive by "
        f"{revised_eta.isoformat()}. We apologise for the delay and appreciate "
        f"your patience.\n\nYou can reply to this email if you have any questions.\n\n"
        f"Kind regards,\nCustomer Care"
    )
    return subject, body