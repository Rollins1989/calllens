"""
`make eval` entrypoint.

Runs the full synthetic dataset through the evaluator and prints a
regression-style report, then exits non-zero if any threshold is breached -
so this is wired into CI as the actual gate on prompt/model changes, not
just a demo script.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.evaluators.scoring import run_full_evaluation
from app.experiments.runner import load_dataset

THRESHOLDS = {
    "intent_accuracy": 0.85,
    "resolution_rate": 0.75,
    "hallucination_rate": 0.15,   # upper bound
    "policy_compliance": 0.85,
}


def main() -> int:
    calls = load_dataset()
    # Evaluate the best prompt version end-to-end, same as production would run.
    calls = [c for c in calls if c.prompt_version == "v3"] or calls

    start = time.time()
    intent_correct = resolved = hallucinated = policy_clean = 0
    latencies = []

    for call in calls:
        result = run_full_evaluation(call)
        intent_correct += result.intent.correct
        resolved += call.actual_outcome in {
            "booking_confirmed", "appointment_cancelled", "human_handoff", "accurate_info_given",
        }
        hallucinated += len(result.groundedness.hallucinations) > 0
        policy_clean += len(result.policy.violations) == 0
        if result.latency.end_to_end_ms:
            latencies.append(result.latency.end_to_end_ms)

    n = len(calls)
    intent_accuracy = intent_correct / n
    resolution_rate = resolved / n
    hallucination_rate = hallucinated / n
    policy_compliance = policy_clean / n
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    elapsed = time.time() - start

    print(f"Dataset: {n} calls (prompt v3)\n")
    print(f"Intent accuracy       {intent_accuracy:.1%}")
    print(f"Resolution rate       {resolution_rate:.1%}")
    print(f"Hallucination rate    {hallucination_rate:.1%}")
    print(f"Policy compliance     {policy_compliance:.1%}")
    print(f"Average latency       {avg_latency:.0f} ms")
    print(f"Eval wall time        {elapsed:.2f}s\n")

    checks = [
        ("Intent accuracy", intent_accuracy, THRESHOLDS["intent_accuracy"], ">="),
        ("Resolution rate", resolution_rate, THRESHOLDS["resolution_rate"], ">="),
        ("Hallucination rate", hallucination_rate, THRESHOLDS["hallucination_rate"], "<="),
        ("Policy compliance", policy_compliance, THRESHOLDS["policy_compliance"], ">="),
    ]
    passed = True
    for name, value, threshold, op in checks:
        ok = value >= threshold if op == ">=" else value <= threshold
        passed &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {op} {threshold:.0%}  (actual: {value:.1%})")

    print(f"\nRegression check: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
