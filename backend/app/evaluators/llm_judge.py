"""
Layer 2 - LLM-as-judge.

Used only for the things that genuinely need semantic understanding:
relevance, clarity, empathy, instruction-following, naturalness. Everything
that CAN be computed deterministically (Layer 1) or checked against a source
of truth (Layer 3) is deliberately kept out of this layer - an LLM judge is
the least reliable and most expensive signal, so it is scoped narrowly
rather than asked to "rate this call 1-10".

The prompt enforces a fixed rubric (1-5 per dimension, with a required
`reason` string) so scores are at least self-consistent and explainable,
rather than a single unexplained number.
"""
from __future__ import annotations

import json

from app.evaluators.deterministic import compute_repetition_penalty, compute_verbosity_penalty
from app.models.schemas import CallRecord, ConversationQualityResult, RubricScore
from app.pipeline.providers import get_llm

_SYSTEM_PROMPT = """You are a strict, consistent QA analyst grading a customer-service voice \
call transcript. Score the AGENT's performance ONLY on the five dimensions below, each on a \
1-5 scale, using this rubric for every dimension:
1 = unrelated / absent, 2 = partially adequate, 3 = acceptable, 4 = strong, 5 = excellent.

Dimensions:
- relevance: did responses directly address what the customer asked?
- clarity: were responses easy to understand, unambiguous?
- empathy: did the agent acknowledge the customer's situation/emotion appropriately?
- instruction_following: did the agent follow the required process/persona?
- naturalness: did the conversation read as natural rather than robotic/templated?

Return ONLY a JSON object of this exact shape, nothing else:
{"relevance": {"score": <int>, "reason": "<one sentence>"},
 "clarity": {"score": <int>, "reason": "<one sentence>"},
 "empathy": {"score": <int>, "reason": "<one sentence>"},
 "instruction_following": {"score": <int>, "reason": "<one sentence>"},
 "naturalness": {"score": <int>, "reason": "<one sentence>"}}
"""


def _format_transcript(call: CallRecord) -> str:
    lines = [f"{t.speaker.value.upper()}: {t.text}" for t in call.transcript]
    return "\n".join(lines)


def evaluate_conversation_quality(call: CallRecord):
    """Returns (ConversationQualityResult, LLMResponse) - the raw LLM response
    is surfaced so the caller can attach real token counts to cost tracking."""
    llm = get_llm()
    user_prompt = (
        f"Scenario: {call.scenario}\nExpected intent: {call.expected_intent}\n\n"
        f"TRANSCRIPT:\n{_format_transcript(call)}"
    )
    response = llm.complete_json(_SYSTEM_PROMPT, user_prompt, max_tokens=500)

    try:
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").split("\n", 1)[-1]
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Fail safe rather than fail loud: a malformed judge response should
        # not crash the whole evaluation pipeline.
        data = {
            k: {"score": 3, "reason": "LLM judge response could not be parsed; defaulted."}
            for k in ["relevance", "clarity", "empathy", "instruction_following", "naturalness"]
        }

    return ConversationQualityResult(
        relevance=RubricScore(**data["relevance"]),
        clarity=RubricScore(**data["clarity"]),
        empathy=RubricScore(**data["empathy"]),
        instruction_following=RubricScore(**data["instruction_following"]),
        naturalness=RubricScore(**data["naturalness"]),
        verbosity_penalty=compute_verbosity_penalty(call),
        repetition_penalty=compute_repetition_penalty(call),
    ), response


def evaluate_conversation_quality_simple(call: CallRecord) -> ConversationQualityResult:
    """Convenience wrapper returning just the result (no raw LLM response)."""
    result, _ = evaluate_conversation_quality(call)
    return result
