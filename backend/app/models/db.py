"""
Persistence layer. SQLite by default (zero setup for a portfolio reviewer),
swap DATABASE_URL for Postgres in production without touching any other code.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Float, Integer, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class CallRow(Base):
    __tablename__ = "calls"

    call_id: Mapped[str] = mapped_column(String, primary_key=True)
    bot_type: Mapped[str] = mapped_column(String, default="appointment_bot")
    scenario: Mapped[str] = mapped_column(String, default="generic")
    model_tag: Mapped[str] = mapped_column(String, default="baseline")
    prompt_version: Mapped[str] = mapped_column(String, default="v1")
    raw_json: Mapped[str] = mapped_column(Text)              # serialized CallRecord
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String, index=True)
    bot_type: Mapped[str] = mapped_column(String)
    overall_score: Mapped[float] = mapped_column(Float)
    resolved: Mapped[int] = mapped_column(Integer, default=0)  # 1/0
    hallucination_count: Mapped[int] = mapped_column(Integer, default=0)
    policy_violation_count: Mapped[int] = mapped_column(Integer, default=0)
    result_json: Mapped[str] = mapped_column(Text)              # serialized EvaluationResult
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ExperimentRunRow(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def dumps(obj) -> str:
    """Helper: dump a pydantic model or dict to a JSON string."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"))
    return json.dumps(obj)
