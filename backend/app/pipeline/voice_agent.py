"""
Version 2 - the voice agent being evaluated.

This is intentionally a small, transparent booking agent (not the subject
of the project - CallLens, the evaluator, is). It exists so the WebSocket
voice demo has something real to talk to: it tracks slot-filling state
(name / date / time), calls a simulated booking tool, and hands the final
transcript to the exact same `run_full_evaluation` pipeline used for the
replayed Version 1 dataset. Swap `complete_text` for a real LLM call
(already wired through `providers.get_llm()`) to make the dialogue fully
model-driven instead of the state machine below.
"""
from __future__ import annotations

import re
import time
import uuid

from app.models.schemas import CallRecord, LatencyTrace, ToolCall, Turn
from app.pipeline.providers import get_llm, get_stt, get_tts

_DATE_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|"
    r"\d{1,2}(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", re.IGNORECASE)
_NAME_RE = re.compile(r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", re.IGNORECASE)


class VoiceAgent:
    """One instance = one in-progress call."""

    def __init__(self, bot_type: str = "appointment_bot", call_id: str | None = None):
        self.call_id = call_id or f"LIVE-{uuid.uuid4().hex[:8]}"
        self.bot_type = bot_type
        self.state = {"name": None, "date": None, "time": None, "confirmed": False}
        self.transcript: list[Turn] = []
        self.tool_calls: list[ToolCall] = []
        self.llm = get_llm()
        self.stt = get_stt()
        self.tts = get_tts()
        self._start = time.time()
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.last_latency = LatencyTrace()

    # --- turn handling -------------------------------------------------- #

    def ingest_audio(self, audio_bytes: bytes) -> dict:
        """Full one-turn loop: audio in -> transcript+reply+audio out."""
        t0 = time.time()
        customer_text, stt_ms = self.stt.transcribe(audio_bytes)
        self.transcript.append(Turn(speaker="customer", text=customer_text,
                                     timestamp_ms=int((time.time() - self._start) * 1000)))

        self._extract_slots(customer_text)
        reply_text, tool_call = self._decide_reply(customer_text)

        # Wrap the deterministic decision through the LLM layer so a real
        # model can be dropped in later without changing this call site.
        llm_resp = self.llm.complete_text(
            system="You are a concise, friendly clinic booking assistant.",
            user=f"Customer said: '{customer_text}'. Planned reply: '{reply_text}'. "
                 f"Rephrase naturally in under 25 words, keep the same meaning.",
        )
        final_reply = reply_text  # keep deterministic reply for demo reliability;
        # real deployments would use `llm_resp.text` here.
        self.llm_input_tokens += llm_resp.input_tokens
        self.llm_output_tokens += llm_resp.output_tokens

        audio_out, tts_ms = self.tts.synthesize(final_reply)
        self.transcript.append(Turn(speaker="agent", text=final_reply,
                                     timestamp_ms=int((time.time() - self._start) * 1000)))
        if tool_call:
            self.tool_calls.append(tool_call)

        end_to_end_ms = int((time.time() - t0) * 1000) + stt_ms + llm_resp.latency_ms + tts_ms
        self.last_latency = LatencyTrace(
            stt_ms=stt_ms, llm_ms=llm_resp.latency_ms, tts_ms=tts_ms,
            time_to_first_response_ms=stt_ms + llm_resp.latency_ms,
            end_to_end_ms=end_to_end_ms,
        )

        return {
            "customer_text": customer_text,
            "agent_text": final_reply,
            "audio_bytes_len": len(audio_out),
            "latency": self.last_latency.model_dump(),
            "tool_call": tool_call.model_dump() if tool_call else None,
            "state": dict(self.state),
        }

    def _extract_slots(self, text: str) -> None:
        if not self.state["name"]:
            m = _NAME_RE.search(text)
            if m:
                self.state["name"] = m.group(1)
        if not self.state["date"]:
            m = _DATE_RE.search(text)
            if m:
                self.state["date"] = m.group(0)
        if not self.state["time"]:
            m = _TIME_RE.search(text)
            if m:
                self.state["time"] = m.group(0)
        if re.search(r"\byes\b|\bcorrect\b|\bconfirm\b", text, re.IGNORECASE) and self.state["date"] and self.state["time"]:
            self.state["confirmed"] = True

    def _decide_reply(self, text: str) -> tuple[str, ToolCall | None]:
        s = self.state
        if not s["name"]:
            return "Sure, I can help. Could I get your name please?", None
        if not s["date"] or not s["time"]:
            return f"Thanks {s['name']}. What date and time would you like?", None
        if not s["confirmed"]:
            return (
                f"I have {s['date']} at {s['time']} for {s['name']}. Shall I confirm this booking?",
                None,
            )
        tool_call = ToolCall(name="confirm_booking", arguments=dict(s), success=True)
        return f"You're all set, {s['name']} - confirmed for {s['date']} at {s['time']}.", tool_call

    # --- finalize --------------------------------------------------------- #

    def finalize(self) -> CallRecord:
        required = ["identify_customer", "collect_date", "collect_time", "confirm_booking"]
        actual_tool_names = {tc.name for tc in self.tool_calls}
        outcome = "booking_confirmed" if "confirm_booking" in actual_tool_names else "booking_incomplete"
        return CallRecord(
            call_id=self.call_id,
            bot_type=self.bot_type,
            scenario="live_voice_call",
            model_tag="live-session",
            prompt_version="live",
            transcript=self.transcript,
            expected_intent="book_appointment",
            predicted_intent="book_appointment",
            expected_outcome="booking_confirmed",
            actual_outcome=outcome,
            required_actions=required,
            actual_tool_calls=self.tool_calls,
            duration_seconds=round(time.time() - self._start, 2),
            latency=self.last_latency,
            llm_input_tokens=self.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens,
        )
