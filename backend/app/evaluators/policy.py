"""
Policy compliance evaluation.

Rules live in `data/policies.yaml` so a reviewer can add a new compliance
rule without touching Python. Each rule pairs a human-readable statement
with a machine-checkable trigger: a regex that should NOT match the agent's
speech (a forbidden phrase), or a "required after trigger" pattern (e.g. any
mention of a payment dispute must be followed, within N turns, by a
transfer-to-human phrase).
"""
from __future__ import annotations

import re

import yaml

from app.config import POLICIES_PATH
from app.models.schemas import CallRecord, PolicyResult, PolicyViolation


def _load_policies() -> list[dict]:
    if not POLICIES_PATH.exists():
        return []
    data = yaml.safe_load(POLICIES_PATH.read_text()) or {}
    return data.get("policies", [])


def evaluate_policy(call: CallRecord) -> PolicyResult:
    policies = _load_policies()
    violations: list[PolicyViolation] = []
    agent_turns = [(i, t.text) for i, t in enumerate(call.transcript) if t.speaker == "agent"]

    for policy in policies:
        pid = policy.get("id", "P???")
        rule = policy.get("rule", "")
        kind = policy.get("type", "forbidden")  # forbidden | required_followup
        pattern = policy.get("pattern")
        severity = policy.get("severity", "medium")

        if kind == "forbidden":
            if not pattern:
                continue
            for idx, text in agent_turns:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append(
                        PolicyViolation(
                            policy_id=pid,
                            rule=rule,
                            severity=severity,
                            evidence=text.strip(),
                            evidence_turn_index=idx,
                        )
                    )

        elif kind == "required_followup":
            trigger_pattern = policy.get("trigger_pattern")
            followup_pattern = policy.get("followup_pattern", pattern)
            window = int(policy.get("window_turns", 4))
            if not trigger_pattern:
                continue
            for i, turn in enumerate(call.transcript):
                if turn.speaker != "customer":
                    continue
                if re.search(trigger_pattern, turn.text, re.IGNORECASE):
                    window_turns = call.transcript[i : i + window]
                    satisfied = any(
                        t.speaker == "agent" and re.search(followup_pattern, t.text, re.IGNORECASE)
                        for t in window_turns
                    )
                    if not satisfied:
                        violations.append(
                            PolicyViolation(
                                policy_id=pid,
                                rule=rule,
                                severity=severity,
                                evidence=f"Trigger at turn {i} ('{turn.text.strip()}') "
                                         f"was not followed by required action within "
                                         f"{window} turns.",
                                evidence_turn_index=i,
                            )
                        )

    weight = {"low": 0.05, "medium": 0.15, "high": 0.3, "critical": 0.5}
    penalty = sum(weight.get(v.severity, 0.15) for v in violations)
    score = max(0.0, round(1.0 - penalty, 4))
    return PolicyResult(violations=violations, score=score)
