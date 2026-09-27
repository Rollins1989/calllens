from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.experiments.runner import load_dataset
from app.evaluators.scoring import run_full_evaluation
from app.models.db import CallRow, EvaluationRow, SessionLocal, dumps
from app.models.schemas import CallRecord

router = APIRouter(prefix="/api/calls", tags=["calls"])


def _ensure_seeded():
    """Lazily load the synthetic dataset into SQLite on first request."""
    session = SessionLocal()
    try:
        existing = session.execute(select(CallRow.call_id)).scalars().all()
        if existing:
            return
        for call in load_dataset():
            session.add(
                CallRow(
                    call_id=call.call_id,
                    bot_type=call.bot_type,
                    scenario=call.scenario,
                    model_tag=call.model_tag,
                    prompt_version=call.prompt_version,
                    raw_json=dumps(call),
                )
            )
        session.commit()
    finally:
        session.close()


@router.get("")
def list_calls(bot_type: str | None = None, scenario: str | None = None, limit: int = 200):
    _ensure_seeded()
    session = SessionLocal()
    try:
        q = select(CallRow)
        if bot_type:
            q = q.where(CallRow.bot_type == bot_type)
        if scenario:
            q = q.where(CallRow.scenario == scenario)
        rows = session.execute(q.order_by(CallRow.call_id).limit(limit)).scalars().all()

        # attach latest evaluation summary if one exists
        eval_by_call = {}
        eval_rows = session.execute(select(EvaluationRow)).scalars().all()
        for e in eval_rows:
            eval_by_call[e.call_id] = e

        out = []
        for r in rows:
            call_data = json.loads(r.raw_json)
            ev = eval_by_call.get(r.call_id)
            out.append(
                {
                    "call_id": r.call_id,
                    "bot_type": r.bot_type,
                    "scenario": r.scenario,
                    "model_tag": r.model_tag,
                    "prompt_version": r.prompt_version,
                    "duration_seconds": call_data.get("duration_seconds"),
                    "overall_score": ev.overall_score if ev else None,
                    "resolved": bool(ev.resolved) if ev else None,
                    "hallucination_count": ev.hallucination_count if ev else None,
                    "policy_violation_count": ev.policy_violation_count if ev else None,
                }
            )
        return {"count": len(out), "calls": out}
    finally:
        session.close()


@router.get("/{call_id}")
def get_call(call_id: str):
    _ensure_seeded()
    session = SessionLocal()
    try:
        row = session.get(CallRow, call_id)
        if not row:
            raise HTTPException(404, f"Call {call_id} not found")
        eval_row = (
            session.execute(select(EvaluationRow).where(EvaluationRow.call_id == call_id))
            .scalars()
            .first()
        )
        return {
            "call": json.loads(row.raw_json),
            "evaluation": json.loads(eval_row.result_json) if eval_row else None,
        }
    finally:
        session.close()


@router.post("/{call_id}/evaluate")
def evaluate_call(call_id: str):
    _ensure_seeded()
    session = SessionLocal()
    try:
        row = session.get(CallRow, call_id)
        if not row:
            raise HTTPException(404, f"Call {call_id} not found")
        call = CallRecord(**json.loads(row.raw_json))
        result = run_full_evaluation(call)

        # upsert evaluation row
        existing = (
            session.execute(select(EvaluationRow).where(EvaluationRow.call_id == call_id))
            .scalars()
            .first()
        )
        resolved = call.actual_outcome in {
            "booking_confirmed",
            "appointment_cancelled",
            "human_handoff",
            "accurate_info_given",
        }
        if existing:
            existing.overall_score = result.score.overall_score
            existing.resolved = int(resolved)
            existing.hallucination_count = len(result.groundedness.hallucinations)
            existing.policy_violation_count = len(result.policy.violations)
            existing.result_json = dumps(result)
        else:
            session.add(
                EvaluationRow(
                    call_id=call_id,
                    bot_type=call.bot_type,
                    overall_score=result.score.overall_score,
                    resolved=int(resolved),
                    hallucination_count=len(result.groundedness.hallucinations),
                    policy_violation_count=len(result.policy.violations),
                    result_json=dumps(result),
                )
            )
        session.commit()
        return json.loads(dumps(result))
    finally:
        session.close()


@router.post("/evaluate-all")
def evaluate_all():
    _ensure_seeded()
    session = SessionLocal()
    try:
        rows = session.execute(select(CallRow)).scalars().all()
        count = 0
        for row in rows:
            call = CallRecord(**json.loads(row.raw_json))
            result = run_full_evaluation(call)
            resolved = call.actual_outcome in {
                "booking_confirmed",
                "appointment_cancelled",
                "human_handoff",
                "accurate_info_given",
            }
            existing = (
                session.execute(select(EvaluationRow).where(EvaluationRow.call_id == row.call_id))
                .scalars()
                .first()
            )
            if existing:
                existing.overall_score = result.score.overall_score
                existing.resolved = int(resolved)
                existing.hallucination_count = len(result.groundedness.hallucinations)
                existing.policy_violation_count = len(result.policy.violations)
                existing.result_json = dumps(result)
            else:
                session.add(
                    EvaluationRow(
                        call_id=row.call_id,
                        bot_type=call.bot_type,
                        overall_score=result.score.overall_score,
                        resolved=int(resolved),
                        hallucination_count=len(result.groundedness.hallucinations),
                        policy_violation_count=len(result.policy.violations),
                        result_json=dumps(result),
                    )
                )
            count += 1
        session.commit()
        return {"evaluated": count}
    finally:
        session.close()
