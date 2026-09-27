"""Force a clean reseed of the SQLite database from the synthetic dataset.

Useful after regenerating the dataset (`make dataset`) or when you want a
fresh demo state before evaluating everything (`make eval` doesn't touch the
DB - only the API layer does).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.experiments.runner import load_dataset
from app.models.db import Base, CallRow, EvaluationRow, SessionLocal, dumps, engine


def main():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        calls = load_dataset()
        for call in calls:
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
        print(f"Seeded {len(calls)} calls into the database.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
