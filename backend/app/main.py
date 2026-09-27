from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import calls, dashboard, experiments, voice
from app.models.db import init_db

app = FastAPI(
    title="CallLens",
    description="LLM-powered voice-call evaluation & QA platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(calls.router)
app.include_router(dashboard.router)
app.include_router(experiments.router)
app.include_router(voice.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "calllens"}


# Serve the static dashboard (frontend/) at the root, so `uvicorn app.main:app`
# is genuinely all you need to run - no separate frontend build/deploy step.
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
