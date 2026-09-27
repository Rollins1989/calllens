# CallLens

**An LLM-powered evaluation platform for voice AI agents.**
CallLens turns call audio (or a call transcript) into a quantitative QA report:
task completion, intent accuracy, policy compliance, hallucination detection,
conversation quality, sentiment, cost, and latency — with every score
explained, not just reported.

It answers one question a lot of "AI voice agent" demos never actually answer:

> **Was the AI voice agent actually good?**

CallLens is the evaluation system. The voice agent it evaluates (a small
booking assistant included for Version 2) is deliberately *not* the point —
swap in any agent, model, or provider and CallLens still tells you how it did.

---

## Why this exists

Most "I built an AI voice bot" portfolio projects stop at *"it works and
sounds nice."* That tells you nothing about whether it's safe, accurate, or
actually completing the task it was built for. CallLens is the missing piece:
a QA/observability layer purpose-built for conversational AI, in the same
spirit as an APM tool for backend services.

---

## What it does

| Dimension | How it's measured |
|---|---|
| **Intent recognition** | Expected vs. predicted intent, exact match |
| **Task completion** | Checklist of required steps (collect field X, call tool Y, confirm) — a real completion percentage, not a vibe |
| **Hallucination / groundedness** | Every factual claim the agent makes (price, hours, availability) is checked against an authoritative knowledge base |
| **Policy compliance** | A YAML rules file checked against the transcript — forbidden phrases, required follow-ups (e.g. "billing disputes must be transferred to a human") |
| **Conversation quality** | LLM-as-judge, scoped to strictly semantic dimensions (relevance, clarity, empathy, instruction-following, naturalness) on a fixed 1–5 rubric — never a bare "8/10" |
| **Sentiment** | Initial → final delta and whether the agent de-escalated, not just positive/negative |
| **Conversation metrics** | Turns, utterance length, interruptions, duration |
| **Latency** | STT / LLM / TTS / time-to-first-response / end-to-end, per call |
| **Cost** | STT + LLM input/output + TTS, broken down per call, rolled up as cost-per-resolution |

Every call gets one explainable **0–100 score**, broken into weighted,
per-dimension points — never a single unexplained number — plus a
plain-English **primary failure** and a concrete **recommendation**.

---

## Architecture

### Version 1 — replay & evaluate (no live audio required)

```
   SYNTHETIC / IMPORTED CALL TRANSCRIPT
                  │
                  ▼
   ┌─────────────────────────────┐
   │        CallRecord            │  (Pydantic contract)
   └──────────────┬───────────────┘
                  │
     ┌────────────┼─────────────────┐
     ▼            ▼                 ▼
 Layer 1:      Layer 2:          Layer 3:
 Deterministic LLM-as-Judge      Reference-based
 (task done?,  (relevance,       (hallucination vs.
  intent OK?,   clarity, empathy, knowledge base;
  metrics)      naturalness)      policy vs. rules.yaml)
     │            │                 │
     └────────────┼─────────────────┘
                  ▼
            Score Engine
      (per-bot-type weighted,
       fully explained)
                  │
                  ▼
            FastAPI API
                  │
                  ▼
         Dashboard (Overview /
      Calls / Experiments tabs)
```

### Version 2 — live voice pipeline

```
  Browser mic
      │
      ▼
   WebSocket  /ws/voice
      │
      ▼
     STT  ───────────────┐
      │                   │  pluggable providers
      ▼                   │  (mock by default, real
   Voice Agent            │   STT/TTS/LLM behind one
   (slot-filling +        │   env var each — see
    booking tool call)    │   pipeline/providers.py)
      │                   │
      ▼                   │
     TTS  ───────────────┘
      │
      ▼
  Browser speaker
      │
      ▼
  Full transcript ──────────────► same CallLens evaluator as Version 1
```

The voice agent under test and the evaluation system are architecturally
separate on purpose — CallLens doesn't care whether the transcript came from
a replayed dataset or a live call, it scores both identically.

### The "killer feature": three independent evaluation layers

```
                     CALL
                      │
                      ▼
                 Transcript
                      │
           ┌──────────┼───────────┐
           ▼          ▼           ▼
     Deterministic  LLM Judge   Reference /
       Metrics      Metrics    Ground Truth
    (tool calls,   (relevance,  (knowledge base +
     regex slot-   clarity,      policy rules)
     filling)       empathy…)
           │          │           │
           └──────────┼───────────┘
                      ▼
                 Score Engine
                      │
                      ▼
               Final QA Report
```

Deterministic checks are cheap, free, and 100% reproducible — they're what
the regression suite (`make eval`) runs on every change. The LLM judge is
scoped narrowly to things that genuinely need semantic understanding, so it's
never asked to do a computer's job. Reference-based checks catch the failure
mode that matters most for a business-facing agent: confidently stating
something false.

---

## Quickstart

```bash
git clone <this-repo>
cd calllens
make install       # pip install -r requirements.txt
make dataset       # generates 90 synthetic labeled calls
make run           # starts the API + dashboard at http://localhost:8000
```

Open **http://localhost:8000** — the dashboard is served directly by the
FastAPI app, no separate frontend build step. Click **"Evaluate all calls"**
on the Calls tab to populate scores, then explore.

No API keys are required to run the full system: the LLM judge, STT and TTS
all default to deterministic **mock providers** so CI, demos, and local
development work out of the box. Copy `.env.example` to `.env` and set
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` to switch the judge to a real model —
no code changes needed.

```bash
cp .env.example .env
# edit .env, then:
make run
```

### Docker

```bash
docker compose up --build
```

---

## Running the evaluation suite

```bash
make test    # 22 unit + integration tests (pytest)
make eval    # dataset-wide regression gate, CI-style output:
```

```
Dataset: 30 calls (prompt v3)

Intent accuracy       100.0%
Resolution rate       93.3%
Hallucination rate    0.0%
Policy compliance     90.0%
Average latency       1708 ms

  [PASS] Intent accuracy >= 85%  (actual: 100.0%)
  [PASS] Resolution rate >= 75%  (actual: 93.3%)
  [PASS] Hallucination rate <= 15%  (actual: 0.0%)
  [PASS] Policy compliance >= 85%  (actual: 90.0%)

Regression check: PASS
```

This is exactly what runs in CI (`.github/workflows/ci.yml`) on every push —
a genuine regression gate on prompt/model changes, not just a demo script.

---

## Experiments: prompt & model comparison

The synthetic dataset replays every one of its 10 scenario types under
**3 prompt versions** and **3 model tags**, so the Experiment tab in the
dashboard (or `POST /api/experiments/run?group_by=prompt_version`) produces a
real, meaningfully different comparison:

| Metric | v1 (baseline) | v2 (structured) | v3 (guarded) |
|---|---|---|---|
| Intent accuracy | 90.0% | 96.7% | 100% |
| Task completion | 75.5% | 77.8% | 80.2% |
| Resolution | 63.3% | 83.3% | 93.3% |
| Hallucination rate | 3.3% | 0% | 0% |
| Policy compliance | 96.7% | 98.3% | 100% |

> v3 vs v1: hallucination rate 3.3% → 0%, task completion 75.5% → 80.2%,
> policy compliance 96.7% → 100.0%.

Swap `group_by=model_tag` for the same comparison across models instead of
prompts — same pipeline, same evaluator, so it's never apples-to-oranges.
See `backend/app/experiments/prompts/` for the actual prompt text behind
each version (v1 is a one-liner; v3 adds explicit process steps and hard
guardrails — which is *why* it scores better).

---

## Project structure

```
calllens/
├── backend/
│   └── app/
│       ├── main.py                 FastAPI app entrypoint
│       ├── config.py               env-driven config, scoring weights, cost table
│       ├── models/
│       │   ├── schemas.py          Pydantic contracts (CallRecord, EvaluationResult…)
│       │   └── db.py               SQLAlchemy models (SQLite by default)
│       ├── evaluators/
│       │   ├── deterministic.py    Layer 1
│       │   ├── llm_judge.py        Layer 2
│       │   ├── groundedness.py     Layer 3 - hallucination detection
│       │   ├── policy.py           Layer 3 - compliance rules
│       │   ├── sentiment.py
│       │   └── scoring.py          combines all layers into one explained score
│       ├── pipeline/
│       │   ├── providers.py        pluggable LLM / STT / TTS (mock + real)
│       │   ├── voice_agent.py      Version 2 booking agent under test
│       │   └── cost_tracker.py
│       ├── api/                    calls.py, dashboard.py, experiments.py, voice.py
│       ├── experiments/
│       │   ├── runner.py           prompt/model comparison engine
│       │   └── prompts/            v1/v2/v3 system prompts
│       └── data/
│           ├── knowledge_base.json source of truth for groundedness checks
│           ├── policies.yaml       compliance rules
│           ├── generate_dataset.py synthetic dataset generator
│           └── synthetic_calls/    90 generated labeled calls (gitignored raw output is regenerable)
│   └── tests/                      22 pytest tests incl. full-dataset regression suite
├── frontend/
│   └── index.html                  single-page dashboard (vanilla JS + Chart.js)
├── scripts/
│   ├── run_eval.py                 `make eval` CLI
│   └── seed_db.py                  force a clean DB reseed
├── .github/workflows/ci.yml
├── Dockerfile / docker-compose.yml
└── Makefile
```

---

## API reference

| Method & path | Description |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/calls` | List calls (filter by `bot_type`, `scenario`) |
| `GET /api/calls/{call_id}` | Full call + evaluation detail |
| `POST /api/calls/{call_id}/evaluate` | Run the full evaluator on one call |
| `POST /api/calls/evaluate-all` | Evaluate the entire dataset |
| `GET /api/dashboard/summary` | Aggregate stats + QA-score-over-time series |
| `POST /api/experiments/run?group_by=` | `prompt_version` \| `model_tag` \| `bot_type` comparison |
| `WS /ws/voice` | Live Version 2 voice loop (see protocol in `api/voice.py`) |

Interactive docs at **`/docs`** (FastAPI's built-in Swagger UI).

---

## Design decisions worth calling out

- **Three separate evaluation layers, not one LLM call.** Anything a
  computer can check with certainty (Layer 1) or against a source of truth
  (Layer 3) is kept out of the LLM judge entirely — the least reliable and
  most expensive signal is scoped as narrowly as possible.
- **Every score is explainable.** `ScoreBreakdown` always carries the
  per-dimension points *and* the weights used to get there, plus a
  `primary_failure` string — never a bare number.
- **Provider-agnostic by construction.** `pipeline/providers.py` defines
  small interfaces for LLM/STT/TTS with a fully-offline mock implementation
  and real implementations behind the same interface — the Experiment
  Runner's model comparison is a first-class feature, not an afterthought.
- **Deterministic-first regression testing.** `make eval` runs the same
  pipeline CI runs, on the same seeded dataset, so a prompt or scoring change
  either passes the same bar or visibly regresses — this is the difference
  between "I built a chatbot" and "I built a system with a QA gate."
- **Config-driven, not hardcoded.** Scoring weights (`config.py`), policy
  rules (`policies.yaml`), and the knowledge base (`knowledge_base.json`) are
  all editable without touching evaluator code.

## What's mocked, and why that's honest

- **LLM judge**: defaults to a deterministic heuristic (`MockLLM`) so the
  system runs and its tests pass with zero API keys / cost / network access.
  Set `LLM_PROVIDER=anthropic` or `openai` + the matching API key for real
  semantic judging — no other code changes.
- **STT/TTS** in the Version 2 voice demo default to mocks that don't decode
  real audio (`MockSTT` returns a scripted demo line each turn) so the full
  loop — mic → agent → evaluator — is demonstrable without any speech
  infrastructure. Real providers (`OpenAIWhisperSTT`, `OpenAITTS`) are wired
  in behind the same interface; point `STT_PROVIDER`/`TTS_PROVIDER` at them
  in `.env` to go fully live.
- Cost figures use an illustrative pricing table (`config.COST_TABLE`) —
  replace with your provider's published rates for real numbers.

---

## Roadmap ideas

- Swap SQLite for Postgres in `docker-compose.yml` for multi-writer production use.
- Add LiveKit/Pipecat as a real telephony front-end to the Version 2 pipeline.
- Persist experiment runs over time to chart prompt-version drift, not just a point-in-time comparison.
- Add a `/api/calls/import` endpoint for uploading real call recordings + a transcription step.

---

## License

MIT — see [LICENSE](LICENSE).
