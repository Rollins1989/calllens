from __future__ import annotations

from fastapi import APIRouter, Query

from app.experiments.runner import run_experiment
from app.models.db import ExperimentRunRow, SessionLocal, dumps

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/run")
def run(group_by: str = Query("prompt_version", pattern="^(prompt_version|model_tag|bot_type)$")):
    result = run_experiment(group_by=group_by)
    session = SessionLocal()
    try:
        session.add(ExperimentRunRow(name=f"compare_by_{group_by}", result_json=dumps(result)))
        session.commit()
    finally:
        session.close()
    return result


@router.get("/prompt-comparison")
def prompt_comparison():
    return run_experiment(group_by="prompt_version")


@router.get("/model-comparison")
def model_comparison():
    return run_experiment(group_by="model_tag")
