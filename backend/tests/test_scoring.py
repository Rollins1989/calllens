from app.evaluators.scoring import run_full_evaluation
from app.models.schemas import CallRecord, ToolCall, Turn


def _good_call():
    return CallRecord(
        call_id="T-20",
        bot_type="appointment_bot",
        transcript=[
            Turn(speaker="customer", text="Hi, my name is Priya Shah, I'd like to book for Tuesday at 4 pm."),
            Turn(speaker="agent", text="Thanks Priya, I have you down for Tuesday at 4 pm, shall I confirm?"),
            Turn(speaker="customer", text="Yes please confirm."),
            Turn(speaker="agent", text="You're all set, confirmed for Tuesday at 4 pm."),
        ],
        expected_intent="book_appointment",
        predicted_intent="book_appointment",
        actual_outcome="booking_confirmed",
        required_actions=["identify_customer", "collect_date", "collect_time", "confirm_booking"],
        actual_tool_calls=[ToolCall(name="confirm_booking", success=True)],
        duration_seconds=40,
    )


def _bad_call():
    return CallRecord(
        call_id="T-21",
        bot_type="support_bot",
        transcript=[
            Turn(speaker="customer", text="I was billed twice, unauthorized charge, refund me now!"),
            Turn(speaker="agent", text="Your refund will be processed right away, no worries."),
        ],
        expected_intent="billing_dispute",
        predicted_intent="book_appointment",
        actual_outcome="unauthorized_refund_promised",
        required_actions=["understand_request", "handoff_to_human"],
        actual_tool_calls=[],
        duration_seconds=20,
    )


def test_good_call_scores_highly():
    result = run_full_evaluation(_good_call())
    assert result.score.overall_score >= 70
    assert result.task_completion.score == 1.0


def test_bad_call_scores_lower_and_flags_primary_failure():
    result = run_full_evaluation(_bad_call())
    good_result = run_full_evaluation(_good_call())
    assert result.score.overall_score < good_result.score.overall_score
    assert result.score.primary_failure is not None
    assert len(result.policy.violations) >= 1


def test_score_breakdown_dimensions_sum_to_overall():
    result = run_full_evaluation(_good_call())
    total_points = sum(result.score.dimension_points.values())
    assert abs(total_points - result.score.overall_score) < 0.01


def test_weights_used_sum_to_one():
    result = run_full_evaluation(_good_call())
    assert abs(sum(result.score.weights_used.values()) - 1.0) < 0.01
