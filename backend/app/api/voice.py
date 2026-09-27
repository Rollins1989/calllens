from __future__ import annotations

import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.evaluators.scoring import run_full_evaluation
from app.models.db import CallRow, EvaluationRow, SessionLocal, dumps
from app.pipeline.voice_agent import VoiceAgent

router = APIRouter(tags=["voice"])


def _persist(call, result) -> None:
    session = SessionLocal()
    try:
        session.merge(
            CallRow(
                call_id=call.call_id,
                bot_type=call.bot_type,
                scenario=call.scenario,
                model_tag=call.model_tag,
                prompt_version=call.prompt_version,
                raw_json=dumps(call),
            )
        )
        session.add(
            EvaluationRow(
                call_id=call.call_id,
                bot_type=call.bot_type,
                overall_score=result.score.overall_score,
                resolved=int(call.actual_outcome == "booking_confirmed"),
                hallucination_count=len(result.groundedness.hallucinations),
                policy_violation_count=len(result.policy.violations),
                result_json=dumps(result),
            )
        )
        session.commit()
    finally:
        session.close()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    """
    Protocol (JSON text frames both ways):
      client -> {"type": "audio_chunk", "audio_b64": "<base64 webm/pcm bytes>"}
      client -> {"type": "end_call"}
      server -> {"type": "agent_turn", ...}          (after each audio_chunk)
      server -> {"type": "call_report", ...}         (after end_call)

    STT/TTS are pluggable (see pipeline/providers.py). With no API keys set,
    MockSTT ignores the actual audio bytes and returns a scripted demo
    utterance each turn, so the full loop - and the evaluator at the end -
    still runs end-to-end with zero configuration.
    """
    await websocket.accept()
    agent = VoiceAgent()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "end_call":
                call = agent.finalize()
                result = run_full_evaluation(call)
                _persist(call, result)
                await websocket.send_text(json.dumps({
                    "type": "call_report",
                    "call_id": call.call_id,
                    "evaluation": json.loads(dumps(result)),
                }))
                break

            if msg.get("type") == "audio_chunk":
                audio_bytes = base64.b64decode(msg.get("audio_b64", "")) if msg.get("audio_b64") else b""
                turn_result = agent.ingest_audio(audio_bytes)
                await websocket.send_text(json.dumps({"type": "agent_turn", **turn_result}))

    except WebSocketDisconnect:
        pass
