from app.evaluators.deterministic import evaluate_intent, evaluate_task_completion
from app.models.schemas import CallRecord, ToolCall, Turn


def test_intent_correct_when_predicted_matches_expected():
    call = CallRecord(
        call_id="T-2", transcript=[Turn(speaker="customer", text="cancel please")],
        expected_intent="cancel_appointment", predicted_intent="cancel_appointment",
    )
    result = evaluate_intent(call)
    assert result.correct is True


def test_intent_incorrect_when_predicted_diverges():
    call = CallRecord(
        call_id="T-3", transcript=[Turn(speaker="customer", text="cancel please")],
        expected_intent="cancel_appointment", predicted_intent="reschedule_appointment",
    )
    result = evaluate_intent(call)
    assert result.correct is False
    assert result.expected == "cancel_appointment"


def test_task_completion_scores_partial_progress():
    call = CallRecord(
        call_id="T-4",
        transcript=[
            Turn(speaker="customer", text="My name is Arjun."),
            Turn(speaker="agent", text="Thanks Arjun, what date works?"),
        ],
        required_actions=["identify_customer", "collect_date", "collect_time", "confirm_booking"],
        actual_tool_calls=[],
    )
    result = evaluate_task_completion(call)
    assert "identify_customer" in result.completed_steps
    assert "collect_time" in result.missing_steps
    assert 0 < result.score < 1


def test_task_completion_full_via_explicit_tool_calls():
    call = CallRecord(
        call_id="T-5",
        transcript=[Turn(speaker="agent", text="confirmed")],
        required_actions=["confirm_booking"],
        actual_tool_calls=[ToolCall(name="confirm_booking", success=True)],
    )
    result = evaluate_task_completion(call)
    assert result.score == 1.0
    assert result.missing_steps == []
