"""
Regression suite: runs the FULL synthetic dataset through the evaluator and
asserts on dataset-level statistics, not individual calls. This is what
`make eval` runs on every prompt/model change - the CI equivalent of the
brief's `$ make eval ... Regression check: PASS` example.

Thresholds are set comfortably below the current baseline (see the printed
experiment comparison) so a real regression trips these tests, while normal
variance from dataset edits does not.
"""
from app.evaluators.scoring import run_full_evaluation
from app.experiments.runner import load_dataset


def test_dataset_loads_with_expected_size():
    calls = load_dataset()
    assert len(calls) >= 50, "Synthetic dataset should contain at least 50 calls"


def test_regression_thresholds_hold_on_best_prompt_version():
    calls = [c for c in load_dataset() if c.prompt_version == "v3"]
    assert calls, "Expected v3 calls in the dataset"

    intent_correct = 0
    resolved = 0
    hallucinated = 0
    policy_clean = 0

    for call in calls:
        result = run_full_evaluation(call)
        intent_correct += result.intent.correct
        resolved += call.actual_outcome in {
            "booking_confirmed", "appointment_cancelled", "human_handoff", "accurate_info_given",
        }
        hallucinated += len(result.groundedness.hallucinations) > 0
        policy_clean += len(result.policy.violations) == 0

    n = len(calls)
    intent_accuracy = intent_correct / n
    resolution_rate = resolved / n
    hallucination_rate = hallucinated / n
    policy_compliance = policy_clean / n

    assert intent_accuracy >= 0.85, f"Intent accuracy regressed: {intent_accuracy:.2%}"
    assert resolution_rate >= 0.75, f"Resolution rate regressed: {resolution_rate:.2%}"
    assert hallucination_rate <= 0.15, f"Hallucination rate regressed: {hallucination_rate:.2%}"
    assert policy_compliance >= 0.85, f"Policy compliance regressed: {policy_compliance:.2%}"


def test_no_call_crashes_the_evaluator():
    for call in load_dataset():
        result = run_full_evaluation(call)
        assert 0 <= result.score.overall_score <= 100
