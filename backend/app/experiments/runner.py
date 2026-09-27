"""
Experiment runner.

Answers two portfolio-grade questions the brief calls out explicitly:

  1. "Prompt v3 reduced hallucination by X points while increasing task
     completion" -> group_by="prompt_version"
  2. Which model performs best on this bot, all else equal?
     -> group_by="model_tag"

Both reuse the exact same evaluation pipeline as a single-call evaluation,
so a comparison is never apples-to-oranges.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from app.config import SYNTHETIC_CALLS_DIR
from app.evaluators.scoring import run_full_evaluation
from app.models.schemas import CallRecord
import json


def load_dataset() -> list[CallRecord]:
    calls = []
    for path in sorted(SYNTHETIC_CALLS_DIR.glob("CALL-*.json")):
        data = json.loads(path.read_text())
        calls.append(CallRecord(**data))
    return calls


def _group_metrics(evaluations: list[dict]) -> dict:
    if not evaluations:
        return {}
    n = len(evaluations)
    return {
        "n_calls": n,
        "intent_accuracy_pct": round(100 * mean(e["intent_correct"] for e in evaluations), 1),
        "task_completion_pct": round(100 * mean(e["task_completion_score"] for e in evaluations), 1),
        "resolution_pct": round(100 * mean(e["resolved"] for e in evaluations), 1),
        "hallucination_rate_pct": round(100 * mean(e["hallucination_count"] > 0 for e in evaluations), 1),
        "policy_compliance_pct": round(100 * mean(e["policy_score"] for e in evaluations), 1),
        "avg_overall_score": round(mean(e["overall_score"] for e in evaluations), 1),
        "avg_latency_ms": round(mean(e["latency_ms"] for e in evaluations), 0),
        "avg_cost_usd": round(mean(e["cost_usd"] for e in evaluations), 4),
    }


RESOLVED_OUTCOMES = {"booking_confirmed", "appointment_cancelled", "human_handoff", "accurate_info_given"}


def _to_row(call: CallRecord, result) -> dict:
    return {
        "call_id": call.call_id,
        "intent_correct": result.intent.correct,
        "task_completion_score": result.task_completion.score,
        "resolved": call.actual_outcome in RESOLVED_OUTCOMES,
        "hallucination_count": len(result.groundedness.hallucinations),
        "policy_score": result.policy.score,
        "overall_score": result.score.overall_score,
        "latency_ms": result.latency.end_to_end_ms or 0,
        "cost_usd": result.cost.total_usd,
    }


def run_experiment(group_by: str = "prompt_version") -> dict:
    assert group_by in ("prompt_version", "model_tag", "bot_type")
    calls = load_dataset()
    grouped_rows: dict[str, list[dict]] = defaultdict(list)

    for call in calls:
        result = run_full_evaluation(call.model_copy(deep=True))
        key = getattr(call, group_by)
        grouped_rows[key].append(_to_row(call, result))

    comparison = {group: _group_metrics(rows) for group, rows in grouped_rows.items()}

    # Headline delta callouts, e.g. "v3 reduced hallucination by 6.0 points
    # vs v1 while increasing task completion by 12.3 points" - computed only
    # for prompt_version since that's the natural "before/after" axis.
    headline = None
    if group_by == "prompt_version" and "v1" in comparison and "v3" in comparison:
        v1, v3 = comparison["v1"], comparison["v3"]
        headline = (
            f"v3 vs v1: hallucination rate {v1['hallucination_rate_pct']}% -> "
            f"{v3['hallucination_rate_pct']}% "
            f"({v3['hallucination_rate_pct'] - v1['hallucination_rate_pct']:+.1f} pts), "
            f"task completion {v1['task_completion_pct']}% -> {v3['task_completion_pct']}% "
            f"({v3['task_completion_pct'] - v1['task_completion_pct']:+.1f} pts), "
            f"policy compliance {v1['policy_compliance_pct']}% -> {v3['policy_compliance_pct']}% "
            f"({v3['policy_compliance_pct'] - v1['policy_compliance_pct']:+.1f} pts)."
        )

    return {"group_by": group_by, "comparison": comparison, "headline": headline}


if __name__ == "__main__":
    import sys
    group = sys.argv[1] if len(sys.argv) > 1 else "prompt_version"
    out = run_experiment(group)
    print(json.dumps(out, indent=2))
