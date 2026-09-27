"""
Central configuration for CallLens.

Everything is overridable via environment variables / .env so the same
code runs in three modes:
  1. Fully offline (mock LLM + mock STT/TTS)   -> used by CI and `make eval`
  2. LLM-judge enabled (real ANTHROPIC/OPENAI key, mock voice)
  3. Full voice pipeline (real STT/TTS keys too)
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_CALLS_DIR = DATA_DIR / "synthetic_calls"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
POLICIES_PATH = DATA_DIR / "policies.yaml"
PROMPTS_DIR = BASE_DIR / "experiments" / "prompts"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR.parent / 'calllens.db'}")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

STT_PROVIDER = os.getenv("STT_PROVIDER", "mock").lower()
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "mock").lower()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

DEFAULT_BOT_TYPE = os.getenv("DEFAULT_BOT_TYPE", "appointment_bot")

# --- Scoring weights, per bot type (must each sum to 1.0) -----------------
SCORING_WEIGHTS = {
    "appointment_bot": {
        "task_completion": 0.35,
        "accuracy": 0.20,
        "policy_compliance": 0.20,
        "conversation_quality": 0.15,
        "customer_experience": 0.10,
    },
    "support_bot": {
        "accuracy": 0.30,
        "task_completion": 0.25,
        "policy_compliance": 0.20,
        "conversation_quality": 0.15,
        "customer_experience": 0.10,
    },
    "default": {
        "task_completion": 0.25,
        "accuracy": 0.20,
        "policy_compliance": 0.20,
        "groundedness": 0.15,
        "conversation_quality": 0.10,
        "customer_experience": 0.10,
    },
}

# Rough cost table, USD, used purely for the cost-tracking dashboard.
# Numbers are illustrative placeholders you should replace with your
# provider's current published pricing.
COST_TABLE = {
    "stt_per_minute": 0.006,
    "tts_per_1k_chars": 0.015,
    "llm_input_per_1k_tokens": 0.003,
    "llm_output_per_1k_tokens": 0.015,
}

LATENCY_SLO_MS = {
    "stt": 300,
    "llm": 800,
    "tts": 300,
    "end_to_end": 1500,
}
