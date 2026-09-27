"""
Sentiment is treated as one signal among many (see README) rather than the
centerpiece of the evaluation. What matters more than "positive/negative" is
the *delta*: did the agent de-escalate an upset customer, or make things
worse.
"""
from __future__ import annotations

from app.models.schemas import CallRecord, SentimentResult

_SCALE = {"angry": -2, "frustrated": -1, "neutral": 0, "positive": 1, "delighted": 2}

_NEGATIVE_WORDS = {"angry", "frustrated", "annoyed", "upset", "unacceptable", "ridiculous", "furious"}
_POSITIVE_WORDS = {"thanks", "thank you", "great", "perfect", "appreciate", "helpful", "awesome"}


def _infer_sentiment(text: str) -> str:
    lowered = text.lower()
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lowered)
    pos = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    if neg > pos and neg >= 2:
        return "angry"
    if neg > pos:
        return "frustrated"
    if pos > neg and pos >= 2:
        return "delighted"
    if pos > neg:
        return "positive"
    return "neutral"


def evaluate_sentiment(call: CallRecord) -> SentimentResult:
    customer_turns = [t.text for t in call.transcript if t.speaker == "customer"]

    initial = call.initial_sentiment or (_infer_sentiment(customer_turns[0]) if customer_turns else "neutral")
    final = call.final_sentiment or (_infer_sentiment(customer_turns[-1]) if customer_turns else "neutral")

    delta = _SCALE.get(final, 0) - _SCALE.get(initial, 0)
    return SentimentResult(
        initial=initial,
        final=final,
        delta=float(delta),
        de_escalated=delta > 0 and _SCALE.get(initial, 0) < 0,
    )
