"""
The Score Engine.

Combines Layer 1 (deterministic), Layer 2 (LLM judge) and Layer 3
(reference/groundedness/policy) into one weighted, per-bot-type,
fully-explained score - never a single unexplained number (see README
"Make it explain WHY the score happened").
"""
from __future__ import annotations

from app.config import SCORING_WEIGHTS
from app.evaluators.deterministic import (
    evaluate_conversation_metrics,
    evaluate_intent,
    evaluate_task_completion,
)
from app.evaluators.groundedness import evaluate_groundedness
from app.evaluators.llm_judge import evaluate_conversation_quality
from app.evaluators.policy import evaluate_policy
from app.evaluators.sentiment import evaluate_sentiment
from app.models.schemas import CallRecord, EvaluationResult, ScoreBreakdown
from app.pipeline.cost_tracker import estimate_cost


def _customer_experience_score(sentiment_result, conv_metrics) -> float:
    """A composite of sentiment delta + interruption rate, standing in for
    "did this feel like a good experience to the human on the phone"."""
    delta_component = (sentiment_result.delta + 2) / 4.0  # normalize -2..2 -> 0..1
    interruption_penalty = min(0.3, conv_metrics.interruption_count * 0.1)
    return max(0.0, round(delta_component - interruption_penalty, 4))


def run_full_evaluation(call: CallRecord) -> EvaluationResult:
    intent = evaluate_intent(call)
    task_completion = evaluate_task_completion(call)
    groundedness = evaluate_groundedness(call)
    policy = evaluate_policy(call)
    conv_metrics = evaluate_conversation_metrics(call)
    conv_quality, llm_response = evaluate_conversation_quality(call)
    sentiment = evaluate_sentiment(call)

    # Merge measured token counts from the judge call itself into the call's
    # own counters so cost tracking reflects the *evaluation* cost too.
    call.llm_input_tokens += llm_response.input_tokens
    call.llm_output_tokens += llm_response.output_tokens
    call.latency.llm_ms = call.latency.llm_ms or llm_response.latency_ms
    cost = estimate_cost(call)

    accuracy_score = 1.0 if intent.correct else (0.5 if intent.expected is None else 0.0)
    customer_experience = _customer_experience_score(sentiment, conv_metrics)

    dimension_scores = {
        "task_completion": task_completion.score,
        "accuracy": accuracy_score,
        "policy_compliance": policy.score,
        "groundedness": groundedness.score,
        "conversation_quality": conv_quality.average,
        "customer_experience": customer_experience,
    }

    weights = SCORING_WEIGHTS.get(call.bot_type, SCORING_WEIGHTS["default"])
    # Re-normalize in case a bot_type's weight table omits a dimension we computed
    # (e.g. appointment_bot has no explicit groundedness weight - fold it into accuracy).
    active_weights = {k: v for k, v in weights.items() if k in dimension_scores}
    if "groundedness" not in active_weights and "groundedness" in dimension_scores:
        # groundedness still counts, folded half-and-half into accuracy weight
        gw = active_weights.get("accuracy", 0) / 2
        active_weights["accuracy"] = active_weights.get("accuracy", 0) - gw
        active_weights["groundedness"] = gw
    total_weight = sum(active_weights.values()) or 1.0
    active_weights = {k: v / total_weight for k, v in active_weights.items()}

    dimension_points = {
        dim: round(dimension_scores[dim] * active_weights.get(dim, 0) * 100, 2)
        for dim in dimension_scores
    }
    overall = round(sum(dimension_points.values()), 2)

    # Primary failure = lowest-scoring weighted dimension, surfaced in plain English.
    primary_failure = None
    worst_dim = min(dimension_points, key=lambda d: dimension_points[d]) if dimension_points else None
    if worst_dim and dimension_scores[worst_dim] < 0.6:
        primary_failure = _explain_failure(worst_dim, call, task_completion, policy, groundedness, intent)

    score = ScoreBreakdown(
        call_id=call.call_id,
        bot_type=call.bot_type,
        weights_used=active_weights,
        dimension_scores=dimension_scores,
        dimension_points=dimension_points,
        overall_score=overall,
        primary_failure=primary_failure,
    )

    recommendation = _recommend(call, task_completion, policy, groundedness, intent, score)

    return EvaluationResult(
        call_id=call.call_id,
        intent=intent,
        task_completion=task_completion,
        groundedness=groundedness,
        policy=policy,
        conversation_metrics=conv_metrics,
        conversation_quality=conv_quality,
        sentiment=sentiment,
        cost=cost,
        latency=call.latency,
        score=score,
        recommendation=recommendation,
    )


def _explain_failure(dim, call, task_completion, policy, groundedness, intent) -> str:
    if dim == "task_completion" and task_completion.missing_steps:
        return f"Agent did not complete: {', '.join(task_completion.missing_steps)}."
    if dim == "policy_compliance" and policy.violations:
        v = policy.violations[0]
        return f"Policy violation ({v.policy_id}): {v.evidence}"
    if dim == "groundedness" and groundedness.hallucinations:
        h = groundedness.hallucinations[0]
        return f"Hallucinated claim: {h.claim} (source of truth: {h.source_value})."
    if dim == "accuracy" and not intent.correct:
        return f"Misclassified intent - expected '{intent.expected}', predicted '{intent.predicted}'."
    return f"Lowest-performing dimension was {dim.replace('_', ' ')}."


def _recommend(call, task_completion, policy, groundedness, intent, score: ScoreBreakdown) -> str:
    tips = []
    if task_completion.missing_steps:
        tips.append(f"Add explicit prompting/tool-checks for: {', '.join(task_completion.missing_steps)}.")
    if policy.violations:
        ids = sorted({v.policy_id for v in policy.violations})
        tips.append(f"Reinforce policies {', '.join(ids)} in the system prompt and add a hard guardrail.")
    if groundedness.hallucinations:
        tips.append("Ground price/hours claims by forcing a knowledge-base tool call before stating facts.")
    if not intent.correct and intent.expected:
        tips.append("Improve intent classification - consider a dedicated intent-routing step before generation.")
    if not tips:
        tips.append("No major issues detected; monitor for regressions on future prompt/model changes.")
    return " ".join(tips)
