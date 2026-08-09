"""Input guardrails: PII minimisation, injection detection, allowlisting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3})?[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Common prompt-injection phrasings found in untrusted free-text fields.
INJECTION_PATTERNS = [
    r"ignore (?:the )?(?:previous|above|prior) instructions",
    r"disregard (?:all|the) (?:above|previous)",
    r"system prompt",
    r"you are now",
    r"reveal (?:your )?(?:secret|api key|prompt)",
    r"act as",
    r"</?(?:system|assistant|instruction)>",
    r"print (?:the )?(?:system|hidden) prompt",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class GuardOutcome:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    redactions: int = 0


def redact_pii(text: str) -> tuple[str, int]:
    n = 0
    def sub(pattern: re.Pattern, token: str, s: str) -> str:
        nonlocal n
        s2, k = pattern.subn(token, s)
        n += k
        return s2
    text = sub(EMAIL_RE, "[REDACTED_EMAIL]", text)
    text = sub(CC_RE, "[REDACTED_CARD]", text)
    text = sub(PHONE_RE, "[REDACTED_PHONE]", text)
    return text, n


def detect_injection(text: str) -> list[str]:
    return [m.group(0) for m in INJECTION_RE.finditer(text or "")]


def sanitize_field(value: str, max_len: int = 2000) -> tuple[str, GuardOutcome]:
    """Redact PII and neutralise injection in a single untrusted field."""
    reasons: list[str] = []
    if value is None:
        return "", GuardOutcome(ok=True)
    value = str(value)[:max_len]
    hits = detect_injection(value)
    if hits:
        reasons.append(f"injection_neutralized:{len(hits)}")
        value = INJECTION_RE.sub("[BLOCKED]", value)
    value, red = redact_pii(value)
    if red:
        reasons.append(f"pii_redacted:{red}")
    return value, GuardOutcome(ok=True, reasons=reasons, redactions=red)


def build_allowlisted_payload(
    record: dict, allowlist: list[str], max_len: int = 2000
) -> tuple[dict, GuardOutcome]:
    """Return only allowlisted fields, each sanitized. Untrusted data is isolated
    as *data*, never merged into instructions by the caller."""
    out: dict = {}
    reasons: list[str] = []
    total_red = 0
    for key in allowlist:
        raw = record.get(key, "")
        clean, oc = sanitize_field(str(raw), max_len=max_len)
        out[key] = clean
        reasons.extend(f"{key}:{r}" for r in oc.reasons)
        total_red += oc.redactions
    return out, GuardOutcome(ok=True, reasons=reasons, redactions=total_red)
