"""
Layer 3 - Reference-based evaluation: groundedness / hallucination detection.

The agent's factual claims (prices, hours, policies) are extracted from the
transcript with targeted regexes and cross-checked against an authoritative
knowledge base (`data/knowledge_base.json`). Anything that contradicts the
source of truth is flagged as a hallucination with a severity.

This is intentionally rule-based rather than another LLM call: for
structured facts like "the fee is X", a regex + lookup is more reliable,
cheaper, and fully reproducible than asking a model to "check for
hallucinations" - which is itself a hallucination-prone task.
"""
from __future__ import annotations

import json
import re

from app.config import KNOWLEDGE_BASE_PATH
from app.models.schemas import CallRecord, GroundednessResult, HallucinationClaim

_CURRENCY_RE = re.compile(r"(?:₹|rs\.?|inr)\s?(\d[\d,]*)", re.IGNORECASE)
_HOURS_RE = re.compile(
    r"\b(\d{1,2})\s*(am|pm)?\s*(?:to|-|–)\s*(\d{1,2})\s*(am|pm)\b", re.IGNORECASE
)
_SUNDAY_CLAIM_RE = re.compile(r"\bsunday[s]?\b[^.]{0,40}\b(open|closed)\b", re.IGNORECASE)


def _load_kb() -> dict:
    if not KNOWLEDGE_BASE_PATH.exists():
        return {}
    return json.loads(KNOWLEDGE_BASE_PATH.read_text())


def _kb_number(value: str) -> float | None:
    if value is None:
        return None
    digits = re.sub(r"[^\d.]", "", str(value))
    return float(digits) if digits else None


def evaluate_groundedness(call: CallRecord) -> GroundednessResult:
    kb = _load_kb()
    claims: list[HallucinationClaim] = []

    for idx, turn in enumerate(call.transcript):
        if turn.speaker != "agent":
            continue
        text = turn.text

        # --- Price / fee claims -------------------------------------------------
        for m in _CURRENCY_RE.finditer(text):
            claimed = _kb_number(m.group(1))
            truth = _kb_number(kb.get("consultation_fee"))
            if truth is not None and claimed is not None and abs(claimed - truth) > 0.01:
                claims.append(
                    HallucinationClaim(
                        claim=f"Agent stated an amount of {m.group(0)}",
                        source_value=str(kb.get("consultation_fee")),
                        severity="high",
                        evidence_turn_index=idx,
                    )
                )

        # --- Sunday open/closed claims -------------------------------------------
        sm = _SUNDAY_CLAIM_RE.search(text)
        if sm:
            claimed_open = sm.group(1).lower() == "open"
            truth_closed = str(kb.get("sunday", "")).lower() == "closed"
            if claimed_open == truth_closed:  # contradicts KB either direction
                claims.append(
                    HallucinationClaim(
                        claim=f"Agent said Sunday is '{sm.group(1)}'",
                        source_value=str(kb.get("sunday")),
                        severity="medium",
                        evidence_turn_index=idx,
                    )
                )

        # --- Operating hours claims ----------------------------------------------
        hm = _HOURS_RE.search(text)
        if hm and kb.get("clinic_hours"):
            claimed_hours = hm.group(0)
            if claimed_hours.replace(" ", "").lower() not in str(
                kb.get("clinic_hours")
            ).replace(" ", "").lower():
                claims.append(
                    HallucinationClaim(
                        claim=f"Agent stated hours as '{claimed_hours}'",
                        source_value=str(kb.get("clinic_hours")),
                        severity="medium",
                        evidence_turn_index=idx,
                    )
                )

        # --- Invented availability (policy P004 overlaps here too) ---------------
        if re.search(r"\bwe have (?:an )?opening (?:at|on) .*guarantee", text, re.IGNORECASE):
            claims.append(
                HallucinationClaim(
                    claim="Agent guaranteed availability without checking the booking system",
                    source_value=None,
                    severity="high",
                    evidence_turn_index=idx,
                )
            )

    # Score: start at 1.0, subtract per-claim severity weight, floor at 0.
    weight = {"low": 0.05, "medium": 0.15, "high": 0.3}
    penalty = sum(weight.get(c.severity, 0.1) for c in claims)
    score = max(0.0, round(1.0 - penalty, 4))

    return GroundednessResult(hallucinations=claims, score=score)
