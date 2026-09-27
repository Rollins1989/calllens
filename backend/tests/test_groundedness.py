from app.evaluators.groundedness import evaluate_groundedness
from app.models.schemas import CallRecord, Turn


def test_hallucinated_fee_detected():
    call = CallRecord(
        call_id="T-10",
        transcript=[
            Turn(speaker="customer", text="How much is a consultation?"),
            Turn(speaker="agent", text="Our consultation fee is Rs 500."),
        ],
    )
    result = evaluate_groundedness(call)
    assert len(result.hallucinations) >= 1
    assert result.score < 1.0


def test_correct_fee_not_flagged():
    call = CallRecord(
        call_id="T-11",
        transcript=[
            Turn(speaker="customer", text="How much is a consultation?"),
            Turn(speaker="agent", text="Our consultation fee is Rs 800."),
        ],
    )
    result = evaluate_groundedness(call)
    assert result.hallucinations == []
    assert result.score == 1.0


def test_incorrect_sunday_claim_detected():
    call = CallRecord(
        call_id="T-12",
        transcript=[
            Turn(speaker="customer", text="Are you open on Sundays?"),
            Turn(speaker="agent", text="Yes, on Sundays we are open."),
        ],
    )
    result = evaluate_groundedness(call)
    assert any("sunday" in h.claim.lower() for h in result.hallucinations)


def test_no_claims_means_fully_grounded():
    call = CallRecord(
        call_id="T-13",
        transcript=[Turn(speaker="agent", text="Sure, happy to help with that.")],
    )
    result = evaluate_groundedness(call)
    assert result.score == 1.0
