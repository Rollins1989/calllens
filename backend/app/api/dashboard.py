from __future__ import annotations

from sqlalchemy import select

from fastapi import APIRouter

from app.models.db import EvaluationRow, SessionLocal

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def summary():
    session = SessionLocal()
    try:
        rows = session.execute(select(EvaluationRow).order_by(EvaluationRow.id)).scalars().all()
        if not rows:
            return {
                "total_calls": 0,
                "resolution_rate_pct": 0,
                "hallucination_rate_pct": 0,
                "avg_qa_score": 0,
                "score_series": [],
            }
        n = len(rows)
        resolved = sum(r.resolved for r in rows)
        with_halluc = sum(1 for r in rows if r.hallucination_count > 0)
        avg_score = sum(r.overall_score for r in rows) / n
        return {
            "total_calls": n,
            "resolution_rate_pct": round(100 * resolved / n, 1),
            "hallucination_rate_pct": round(100 * with_halluc / n, 1),
            "avg_qa_score": round(avg_score, 1),
            "score_series": [{"call_id": r.call_id, "score": r.overall_score} for r in rows],
        }
    finally:
        session.close()
