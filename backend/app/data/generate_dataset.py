"""
Synthetic dataset generator.

Produces labeled calls across the 10 scenario types called out in the
project brief (successful booking, wrong date, hallucinated price, angry
customer, language switch, interruption, incomplete info, policy violation,
successful transfer, tool failure), each replayed under three prompt
versions (v1/v2/v3, each measurably better than the last) and three model
tags, so the Experiment Runner and Model Comparison features in the
dashboard have real, meaningfully different data to show off.

Run: `python -m app.data.generate_dataset` (or `make dataset`).
Deterministic (seeded) so CI and `make eval` always see the same numbers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from app.config import SYNTHETIC_CALLS_DIR

random.seed(42)

PROMPT_VERSIONS = ["v1", "v2", "v3"]
MODEL_TAGS = ["model-a-8b", "model-b-mini", "model-c-flagship"]

# Quality knobs per (prompt_version, model_tag): higher is better, [0,1].
# v3 + flagship should look clearly best on the dashboard; v1 + 8b clearly worst.
def _quality(prompt_version: str, model_tag: str) -> float:
    pv_bonus = {"v1": 0.0, "v2": 0.18, "v3": 0.34}[prompt_version]
    model_bonus = {"model-a-8b": 0.0, "model-b-mini": 0.14, "model-c-flagship": 0.28}[model_tag]
    return min(0.95, 0.35 + pv_bonus + model_bonus)


def roll(p_success: float) -> bool:
    return random.random() < p_success


def turn(speaker, text, interrupted=False, t=0):
    return {"speaker": speaker, "text": text, "timestamp_ms": t, "interrupted": interrupted}


def base_latency(quality: float):
    # Better systems tend to also be faster-tuned/cached; small effect only.
    jitter = random.randint(-60, 60)
    return {
        "stt_ms": max(80, int(260 - quality * 80) + jitter),
        "llm_ms": max(200, int(950 - quality * 300) + jitter),
        "tts_ms": max(80, int(260 - quality * 60) + jitter),
        "time_to_first_response_ms": max(300, int(1400 - quality * 400) + jitter),
        "end_to_end_ms": max(500, int(2200 - quality * 600) + jitter),
    }


# --------------------------------------------------------------------------- #
# Scenario builders. Each returns a dict matching CallRecord's shape.
# --------------------------------------------------------------------------- #

def scn_successful_booking(idx, pv, model, q):
    good = roll(q)
    transcript = [
        turn("customer", "Hi, I'd like to book an appointment for next Tuesday."),
        turn("agent", "Sure, I can help with that. Can I get your name please?"),
        turn("customer", "My name is Priya Shah, phone number 9876543210."),
        turn("agent", "Thanks Priya. What time on Tuesday works for you?"),
        turn("customer", "Around 4 PM works for me."),
        turn("agent", "Great, I have you down for Tuesday at 4 PM. Shall I confirm this booking?" if good
                       else "Ok let me note that down."),
        turn("customer", "Yes, that's correct, please confirm it."),
        turn("agent", "You're all set - your appointment is confirmed for Tuesday at 4 PM." if good
                       else "Alright, noted."),
    ]
    return dict(
        scenario="successful_booking",
        expected_intent="book_appointment",
        predicted_intent="book_appointment",
        expected_outcome="booking_confirmed",
        actual_outcome="booking_confirmed" if good else "booking_unconfirmed",
        required_actions=["identify_customer", "collect_date", "collect_time", "attempt_booking", "confirm_booking"],
        actual_tool_calls=[{"name": "attempt_booking", "arguments": {}, "success": True}]
        + ([{"name": "confirm_booking", "arguments": {}, "success": True}] if good else []),
        transcript=transcript,
        duration_seconds=95,
        initial_sentiment="neutral",
        final_sentiment="positive" if good else "neutral",
    )


def scn_wrong_date(idx, pv, model, q):
    mixed_up = not roll(q)
    transcript = [
        turn("customer", "I need to cancel my appointment on the 14th."),
        turn("agent", "I understand you want to reschedule your appointment. What new date works?" if mixed_up
                       else "Got it, I'll cancel your appointment on the 14th."),
        turn("customer", "No, I want to cancel it entirely, not reschedule."),
        turn("agent", "My apologies for the confusion - I've cancelled your appointment on the 14th."),
    ]
    return dict(
        scenario="wrong_date_intent_confusion",
        expected_intent="cancel_appointment",
        predicted_intent="reschedule_appointment" if mixed_up else "cancel_appointment",
        expected_outcome="appointment_cancelled",
        actual_outcome="appointment_cancelled",
        required_actions=["identify_customer", "understand_request", "attempt_booking"],
        actual_tool_calls=[{"name": "attempt_booking", "arguments": {"action": "cancel"}, "success": True}],
        transcript=transcript,
        duration_seconds=48,
        initial_sentiment="neutral",
        final_sentiment="frustrated" if mixed_up else "neutral",
    )


def scn_hallucinated_price(idx, pv, model, q):
    hallucinate = not roll(q + 0.1)
    transcript = [
        turn("customer", "How much is a consultation?"),
        turn("agent", "Our consultation fee is ₹500." if hallucinate else "Our consultation fee is ₹800."),
        turn("customer", "And are you open on Sundays?"),
        turn("agent", "Yes we're open on Sundays too." if hallucinate else "No, we're closed on Sundays."),
    ]
    return dict(
        scenario="hallucinated_price",
        expected_intent="general_inquiry",
        predicted_intent="general_inquiry",
        expected_outcome="accurate_info_given",
        actual_outcome="inaccurate_info_given" if hallucinate else "accurate_info_given",
        required_actions=["understand_request"],
        actual_tool_calls=[],
        transcript=transcript,
        duration_seconds=30,
        initial_sentiment="neutral",
        final_sentiment="neutral",
    )


def scn_angry_customer(idx, pv, model, q):
    escalated = roll(q)
    transcript = [
        turn("customer", "This is unacceptable, I've been waiting for 40 minutes and no one called me back!"),
        turn("agent", "I completely understand your frustration, and I'm sorry for the delay. Let me get this "
                      "sorted for you right away." if escalated else "Okay, let me check that."),
        turn("customer", "I want to speak to a manager right now.", interrupted=not escalated),
        turn("agent", "Of course - I'm transferring you to a specialist immediately." if escalated
                       else "I can help you with most things, what's the issue?"),
    ]
    return dict(
        scenario="angry_customer_escalation",
        expected_intent="complaint",
        predicted_intent="complaint",
        expected_outcome="human_handoff",
        actual_outcome="human_handoff" if escalated else "unresolved",
        required_actions=["understand_request", "handoff_to_human"],
        actual_tool_calls=[{"name": "handoff_to_human", "arguments": {}, "success": True}] if escalated else [],
        transcript=transcript,
        duration_seconds=52,
        initial_sentiment="angry",
        final_sentiment="neutral" if escalated else "angry",
    )


def scn_language_switch(idx, pv, model, q):
    handled = roll(q)
    transcript = [
        turn("customer", "Hi, mujhe appointment book karni hai kal ke liye."),
        turn("agent", "Bilkul, main aapki madad kar sakta hoon. Kal kis time appointment chahiye?" if handled
                       else "Sorry, could you say that in English please?"),
        turn("customer", "Kal shaam 5 baje."),
        turn("agent", "Theek hai, maine kal shaam 5 baje ke liye appointment book kar di hai." if handled
                       else "I'm not able to understand, can you repeat in English?"),
    ]
    return dict(
        scenario="customer_language_switch",
        expected_intent="book_appointment",
        predicted_intent="book_appointment" if handled else "unclear",
        expected_outcome="booking_confirmed",
        actual_outcome="booking_confirmed" if handled else "unresolved",
        required_actions=["understand_request", "collect_time", "attempt_booking"],
        actual_tool_calls=[{"name": "attempt_booking", "arguments": {}, "success": True}] if handled else [],
        transcript=transcript,
        duration_seconds=40,
        initial_sentiment="neutral",
        final_sentiment="positive" if handled else "frustrated",
    )


def scn_interruption(idx, pv, model, q):
    handled_well = roll(q)
    transcript = [
        turn("agent", "So to confirm, I'll be booking you for Wednesday at—"),
        turn("customer", "Wait, actually can we make it Thursday instead?", interrupted=True),
        turn("agent", "Of course, let's change that to Thursday. What time works?" if handled_well
                       else "As I was saying, Wednesday at 3 PM is confirmed."),
        turn("customer", "Thursday at 2 PM please." if handled_well else "No, I said Thursday!", interrupted=not handled_well),
        turn("agent", "Got it, Thursday at 2 PM - confirmed." if handled_well else "Apologies, let me correct that to Thursday."),
    ]
    return dict(
        scenario="customer_interruption",
        expected_intent="book_appointment",
        predicted_intent="book_appointment",
        expected_outcome="booking_confirmed",
        actual_outcome="booking_confirmed",
        required_actions=["collect_date", "collect_time", "confirm_booking"],
        actual_tool_calls=[{"name": "confirm_booking", "arguments": {}, "success": True}],
        transcript=transcript,
        duration_seconds=35,
        initial_sentiment="neutral",
        final_sentiment="positive" if handled_well else "frustrated",
    )


def scn_incomplete_info(idx, pv, model, q):
    followed_up = roll(q)
    transcript = [
        turn("customer", "I'd like to book an appointment."),
        turn("agent", "Sure! Could I get your name and a preferred date and time?" if followed_up
                       else "Great, you're booked in."),
        turn("customer", "Sure, it's Arjun." if followed_up else "Okay thanks."),
        turn("agent", "Thanks Arjun, and what date and time would you like?" if followed_up
                       else "Perfect, see you soon."),
    ]
    return dict(
        scenario="incomplete_information",
        expected_intent="book_appointment",
        predicted_intent="book_appointment",
        expected_outcome="booking_confirmed",
        actual_outcome="booking_confirmed" if followed_up else "booking_incomplete",
        required_actions=["identify_customer", "collect_date", "collect_time", "confirm_booking"],
        actual_tool_calls=[{"name": "confirm_booking", "arguments": {}, "success": True}] if followed_up else [],
        transcript=transcript,
        duration_seconds=38,
        initial_sentiment="neutral",
        final_sentiment="neutral",
    )


def scn_policy_violation(idx, pv, model, q):
    violates = not roll(q + 0.15)
    transcript = [
        turn("customer", "I was charged twice for my last visit, I want a refund."),
        turn("agent", "I'm sorry about that! Your refund will be processed right away, no worries." if violates
                       else "I'm sorry to hear that. I'll transfer you to a billing specialist who can verify and process this."),
        turn("customer", "Okay, thank you."),
    ]
    return dict(
        scenario="policy_violation_refund",
        expected_intent="billing_dispute",
        predicted_intent="billing_dispute",
        expected_outcome="human_handoff",
        actual_outcome="unauthorized_refund_promised" if violates else "human_handoff",
        required_actions=["understand_request", "handoff_to_human"],
        actual_tool_calls=[{"name": "handoff_to_human", "arguments": {}, "success": True}] if not violates else [],
        transcript=transcript,
        duration_seconds=33,
        initial_sentiment="frustrated",
        final_sentiment="neutral",
    )


def scn_successful_transfer(idx, pv, model, q):
    transcript = [
        turn("customer", "I have a complex insurance question that's outside a normal booking."),
        turn("agent", "That's a great question for our insurance specialist - transferring you now, please hold."),
        turn("customer", "Thanks."),
    ]
    return dict(
        scenario="successful_transfer",
        expected_intent="insurance_inquiry",
        predicted_intent="insurance_inquiry",
        expected_outcome="human_handoff",
        actual_outcome="human_handoff",
        required_actions=["understand_request", "handoff_to_human"],
        actual_tool_calls=[{"name": "handoff_to_human", "arguments": {}, "success": True}],
        transcript=transcript,
        duration_seconds=20,
        initial_sentiment="neutral",
        final_sentiment="neutral",
    )


def scn_tool_failure(idx, pv, model, q):
    recovered = roll(q)
    transcript = [
        turn("customer", "I'd like to book for Friday at noon."),
        turn("agent", "Let me check availability... I'm sorry, our booking system is temporarily down."),
        turn("agent", "I've noted your details and I'll transfer you to a human agent who can complete the booking manually."
                       if recovered else "Please try calling back later."),
        turn("customer", "Okay." if recovered else "That's not helpful."),
    ]
    return dict(
        scenario="tool_failure_recovery",
        expected_intent="book_appointment",
        predicted_intent="book_appointment",
        expected_outcome="human_handoff",
        actual_outcome="human_handoff" if recovered else "unresolved",
        required_actions=["collect_date", "collect_time", "attempt_booking", "detect_failure", "handoff_to_human"],
        actual_tool_calls=[{"name": "attempt_booking", "arguments": {}, "success": False}]
        + ([{"name": "handoff_to_human", "arguments": {}, "success": True}] if recovered else []),
        transcript=transcript,
        duration_seconds=42,
        initial_sentiment="neutral",
        final_sentiment="neutral" if recovered else "frustrated",
    )


SCENARIOS = [
    scn_successful_booking,
    scn_wrong_date,
    scn_hallucinated_price,
    scn_angry_customer,
    scn_language_switch,
    scn_interruption,
    scn_incomplete_info,
    scn_policy_violation,
    scn_successful_transfer,
    scn_tool_failure,
]


def generate(replicas_per_combo: int = 1) -> list[dict]:
    calls = []
    counter = 1
    for scenario_fn in SCENARIOS:
        for pv in PROMPT_VERSIONS:
            for model in MODEL_TAGS:
                for _ in range(replicas_per_combo):
                    q = _quality(pv, model)
                    call = scenario_fn(counter, pv, model, q)
                    call_id = f"CALL-{counter:03d}"
                    call.update(
                        call_id=call_id,
                        bot_type="appointment_bot" if "policy" not in call["scenario"] else "support_bot",
                        model_tag=model,
                        prompt_version=pv,
                        latency=base_latency(q),
                        llm_input_tokens=random.randint(120, 260),
                        llm_output_tokens=random.randint(60, 180),
                    )
                    calls.append(call)
                    counter += 1
    return calls


def main():
    SYNTHETIC_CALLS_DIR.mkdir(parents=True, exist_ok=True)
    calls = generate(replicas_per_combo=1)
    for call in calls:
        path = SYNTHETIC_CALLS_DIR / f"{call['call_id']}.json"
        path.write_text(json.dumps(call, indent=2))
    manifest = {"count": len(calls), "call_ids": [c["call_id"] for c in calls]}
    (SYNTHETIC_CALLS_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(calls)} synthetic calls in {SYNTHETIC_CALLS_DIR}")


if __name__ == "__main__":
    main()
