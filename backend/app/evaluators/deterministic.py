"""
Layer 1 - Deterministic evaluation.

No LLM calls here on purpose. These are checks a computer can answer with
certainty: was a required tool called, did the transcript contain the
expected fields, how long did the call take, how many interruptions were
there. This layer is fast, free, 100% reproducible, and forms the backbone
of the regression test suite (`make eval`).
"""
from __future__ import annotations

import re

from app.models.schemas import (
    CallRecord,
    ConversationMetrics,
    IntentResult,
    TaskCompletionResult,
)

# Canonical "did the agent collect this field" detectors. Real systems would
# hook these into the agent's own structured tool-call arguments; we also
# regex the transcript so the evaluator still works for pure-text transcripts
# that never called a tool (e.g. the agent forgot to).
FIELD_PATTERNS = {
    "collect_name": r"\bmy name is\b|\bthis is\b [A-Z][a-z]+|\bcalled\b [A-Z][a-z]+",
    "collect_phone": r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    "collect_date": r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                     r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
                     r"aug(ust)?|sep(t)?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
                     r"tomorrow|today|\d{1,2}(st|nd|rd|th)?\s*(am|pm)?)\b",
    "collect_time": r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    "confirm_booking": r"\bconfirm(ed)?\b|\byou'?re all set\b|\bbooked\b|\bscheduled\b",
    "identify_customer": r"\bmy name is\b|\baccount number\b|\bdate of birth\b|\bcan you confirm your\b",
    "understand_request": r".+",  # any customer turn present == request was voiced
    "attempt_booking": r"\bbook(ing)?\b|\bschedule\b|\breserve\b",
    "detect_failure": r"\bsystem is down\b|\bunable to\b|\bcan'?t process\b|\berror\b",
    "handoff_to_human": r"\btransfer(ring)?\b|\bhuman agent\b|\bspecialist\b|\bescalat",
}


def _agent_text(call: CallRecord) -> str:
    return " \n".join(t.text for t in call.transcript if t.speaker == "agent").lower()


def _customer_text(call: CallRecord) -> str:
    return " \n".join(t.text for t in call.transcript if t.speaker == "customer").lower()


def evaluate_intent(call: CallRecord) -> IntentResult:
    expected = call.expected_intent
    predicted = call.predicted_intent
    return IntentResult(
        expected=expected,
        predicted=predicted,
        correct=bool(expected) and expected == predicted,
    )


def evaluate_task_completion(call: CallRecord) -> TaskCompletionResult:
    """Checks each `required_action` step against: (a) an explicit ToolCall
    record with that name, or (b) a transcript pattern match for known
    field-collection steps. Falls back gracefully for unknown step names by
    treating an explicit successful ToolCall as the only evidence."""
    full_text = _agent_text(call) + " " + _customer_text(call)
    tool_call_names = {tc.name for tc in call.actual_tool_calls if tc.success}

    completed = []
    for step in call.required_actions:
        if step in tool_call_names:
            completed.append(step)
            continue
        pattern = FIELD_PATTERNS.get(step)
        if pattern and re.search(pattern, full_text, flags=re.IGNORECASE):
            completed.append(step)

    missing = [s for s in call.required_actions if s not in completed]
    score = len(completed) / len(call.required_actions) if call.required_actions else 1.0
    return TaskCompletionResult(
        required_steps=call.required_actions,
        completed_steps=completed,
        missing_steps=missing,
        score=round(score, 4),
    )


def evaluate_conversation_metrics(call: CallRecord) -> ConversationMetrics:
    customer_turns = [t for t in call.transcript if t.speaker == "customer"]
    agent_turns = [t for t in call.transcript if t.speaker == "agent"]
    interruptions = sum(1 for t in call.transcript if t.interrupted)

    def avg_len(turns) -> float:
        if not turns:
            return 0.0
        return round(sum(len(t.text.split()) for t in turns) / len(turns), 2)

    return ConversationMetrics(
        total_turns=len(call.transcript),
        customer_turns=len(customer_turns),
        agent_turns=len(agent_turns),
        avg_customer_utterance_len=avg_len(customer_turns),
        avg_agent_utterance_len=avg_len(agent_turns),
        interruption_count=interruptions,
        call_duration_seconds=call.duration_seconds,
    )


def compute_repetition_penalty(call: CallRecord) -> float:
    """Penalize the agent for saying near-identical things repeatedly - a
    classic voice-agent failure mode (looping because it didn't track state)."""
    agent_lines = [t.text.strip().lower() for t in call.transcript if t.speaker == "agent"]
    if len(agent_lines) < 2:
        return 0.0
    seen = {}
    for line in agent_lines:
        key = " ".join(line.split()[:6])  # first few words as a fingerprint
        seen[key] = seen.get(key, 0) + 1
    repeats = sum(c - 1 for c in seen.values() if c > 1)
    return round(min(0.2, repeats * 0.05), 4)


def compute_verbosity_penalty(call: CallRecord) -> float:
    agent_turns = [t for t in call.transcript if t.speaker == "agent"]
    if not agent_turns:
        return 0.0
    avg_words = sum(len(t.text.split()) for t in agent_turns) / len(agent_turns)
    # Voice agents should be concise; penalize very long-winded responses.
    if avg_words <= 40:
        return 0.0
    return round(min(0.15, (avg_words - 40) / 400), 4)
