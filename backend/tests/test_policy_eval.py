from app.evaluators.policy import evaluate_policy
from app.models.schemas import CallRecord, Turn


def test_forbidden_refund_promise_flagged():
    call = CallRecord(
        call_id="T-6",
        transcript=[
            Turn(speaker="customer", text="I was billed twice, I want a refund."),
            Turn(speaker="agent", text="Your refund will be processed right away."),
        ],
    )
    result = evaluate_policy(call)
    ids = {v.policy_id for v in result.violations}
    assert "P002" in ids
    assert result.score < 1.0


def test_billing_dispute_without_transfer_flagged():
    call = CallRecord(
        call_id="T-7",
        transcript=[
            Turn(speaker="customer", text="This looks like an unauthorized charge."),
            Turn(speaker="agent", text="Let me look into that for you."),
            Turn(speaker="agent", text="I see the charge, it looks fine to me."),
        ],
    )
    result = evaluate_policy(call)
    ids = {v.policy_id for v in result.violations}
    assert "P003" in ids


def test_billing_dispute_with_prompt_transfer_passes():
    call = CallRecord(
        call_id="T-8",
        transcript=[
            Turn(speaker="customer", text="This is an unauthorized charge on my card."),
            Turn(speaker="agent", text="I understand, transferring you to a billing specialist now."),
        ],
    )
    result = evaluate_policy(call)
    ids = {v.policy_id for v in result.violations}
    assert "P003" not in ids


def test_clean_call_has_perfect_policy_score():
    call = CallRecord(
        call_id="T-9",
        transcript=[
            Turn(speaker="customer", text="Hi, I'd like to book an appointment."),
            Turn(speaker="agent", text="Sure! What date works for you?"),
        ],
    )
    result = evaluate_policy(call)
    assert result.violations == []
    assert result.score == 1.0
