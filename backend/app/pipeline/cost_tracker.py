"""
Per-call cost tracking. Every call is broken down into STT / LLM-in /
LLM-out / TTS cost so the dashboard can show a genuinely useful business
metric: cost per successful resolution, not just "the AI answered".

Pricing in config.COST_TABLE is illustrative - point it at your provider's
actual published rates for real numbers.
"""
from __future__ import annotations

from app.config import COST_TABLE
from app.models.schemas import CallRecord, CostBreakdown


def estimate_cost(call: CallRecord, tts_char_count: int | None = None) -> CostBreakdown:
    minutes = call.duration_seconds / 60.0
    stt_cost = minutes * COST_TABLE["stt_per_minute"]

    llm_in_cost = (call.llm_input_tokens / 1000.0) * COST_TABLE["llm_input_per_1k_tokens"]
    llm_out_cost = (call.llm_output_tokens / 1000.0) * COST_TABLE["llm_output_per_1k_tokens"]

    if tts_char_count is None:
        tts_char_count = sum(len(t.text) for t in call.transcript if t.speaker == "agent")
    tts_cost = (tts_char_count / 1000.0) * COST_TABLE["tts_per_1k_chars"]

    return CostBreakdown(
        stt_usd=round(stt_cost, 6),
        llm_input_usd=round(llm_in_cost, 6),
        llm_output_usd=round(llm_out_cost, 6),
        tts_usd=round(tts_cost, 6),
    )
