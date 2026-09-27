# CallLens

### AI Call Evaluation & Quality Assurance Platform

CallLens is an end-to-end evaluation and observability platform for conversational and voice AI agents.

Instead of asking only **"Does the agent work?"**, CallLens measures whether an agent is:

- completing the requested task
- recognizing the correct intent
- staying grounded in known facts
- following business policies
- producing useful, natural responses
- de-escalating conversations when appropriate
- meeting latency and cost expectations

Every evaluated call receives an **explainable 0–100 QA score**, a per-dimension breakdown, a primary failure reason, and a concrete recommendation.

> **The voice agent is not the product. The evaluation system is.**

---

## Why CallLens?

Many conversational-AI demos stop at a successful conversation.

That is not enough for a production system.

A useful AI QA layer needs to answer questions such as:

- Did the agent actually resolve the customer's request?
- Did it identify the correct intent?
- Did it invent information?
- Did it violate a business rule?
- Was the response relevant and clear?
- Did the customer's sentiment improve?
- How much did the interaction cost?
- How long did the pipeline take?
- Did a prompt/model change make the system better or worse?

CallLens turns those questions into **repeatable, testable evaluation signals**.

---

## Key Results

The current 30-call regression benchmark for **prompt v3** produces:

| Metric | Result | Regression threshold |
|---|---:|---:|
| Intent accuracy | **100.0%** | ≥ 85% |
| Resolution rate | **93.3%** | ≥ 75% |
| Hallucination rate | **0.0%** | ≤ 15% |
| Policy compliance | **90.0%** | ≥ 85% |
| Average latency | **1708 ms** | — |

**Regression check: PASS**

The repository also contains a backend test suite with **22 passing tests**.

> These are results from the included synthetic benchmark and should not be interpreted as production performance on real customer calls.

---

## What It Evaluates

| Evaluation dimension | Implementation |
|---|---|
| **Intent recognition** | Expected vs. predicted intent |
| **Task completion** | Required-step/checklist completion |
| **Resolution** | Whether the customer's requested outcome was achieved |
| **Hallucination / groundedness** | Factual claims checked against an authoritative knowledge base |
| **Policy compliance** | Transcript checked against configurable YAML rules |
| **Conversation quality** | LLM-as-judge rubric covering relevance, clarity, empathy, naturalness and instruction-following |
| **Sentiment** | Initial → final sentiment and de-escalation signal |
| **Conversation metrics** | Turns, utterance length, interruptions and duration |
| **Latency** | STT, LLM, TTS, time-to-first-response and end-to-end latency |
| **Cost** | STT, LLM input/output and TTS cost breakdown |

The final score is not a black-box number. CallLens retains the individual dimension scores, weights, failure explanation and recommendation.

---

## Architecture

### Evaluation pipeline

```text
                  CALL / TRANSCRIPT
                         |
                         v
                +------------------+
                |   CallRecord     |
                | Pydantic schema  |
                +--------+---------+
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
        Deterministic  LLM Judge   Reference
          Checks        Checks      Checks
             |           |           |
             |           |           |
        task/intent   relevance    knowledge
        metrics      clarity      base
        completion   empathy      policies
             |           |           |
             +-----------+-----------+
                         |
                         v
                  +-------------+
                  | Score Engine|
                  +------+------+
                         |
                         v
                 +---------------+
                 | FastAPI API   |
                 +-------+-------+
                         |
                         v
                 +---------------+
                 | Web Dashboard |
                 | Overview      |
                 | Calls         |
                 | Experiments   |
                 +---------------+
```

### Live voice pipeline

```text
 Browser microphone
        |
        v
 WebSocket /ws/voice
        |
        v
      STT
        |
        v
 Voice Agent
 (slot filling +
 booking tool call)
        |
        v
      TTS
        |
        v
 Browser speaker
        |
        v
 Full transcript
        |
        +----------------------+
                               |
                               v
                       CallLens Evaluator
```

The **voice agent and evaluation system are intentionally separated**. A transcript can come from the included replay dataset or from the live voice pipeline and still pass through the same evaluation architecture.

---

## Three-Layer Evaluation Strategy

One of the core design decisions in CallLens is that **not everything should be judged by an LLM**.

### Layer 1 — Deterministic evaluation

Used where exact logic is possible:

- expected vs. predicted intent
- required task steps
- tool-call outcomes
- structured conversation metrics
- regression thresholds

These checks are cheap, reproducible and suitable for CI.

### Layer 2 — Semantic evaluation

An LLM judge is used only for dimensions that require semantic understanding:

- relevance
- clarity
- empathy
- instruction-following
- naturalness

The judge uses a fixed rubric rather than producing an unexplained overall score.

### Layer 3 — Reference-based evaluation

Business-critical claims are checked against explicit sources of truth:

- knowledge base
- policy rules
- required follow-ups
- prohibited behavior

This layer is designed to catch the failure mode that matters heavily for customer-facing AI: **confidently saying something unsupported or incorrect**.

---

## Prompt & Model Experiments

CallLens includes an experiment runner for comparing prompts and model tags against the same synthetic evaluation set.

### Prompt versions

| Metric | v1 Baseline | v2 Structured | v3 Guarded |
|---|---:|---:|---:|
| Intent accuracy | 90.0% | 96.7% | **100.0%** |
| Task completion | 75.5% | 77.8% | **80.2%** |
| Resolution | 63.3% | 83.3% | **93.3%** |
| Hallucination rate | 3.3% | 0.0% | **0.0%** |
| Policy compliance | 96.7% | 98.3% | **100.0%** |

The same evaluation pipeline can compare:

- `prompt_version`
- `model_tag`
- `bot_type`

This makes prompt/model changes measurable instead of relying on subjective demo quality.

---

## Dashboard

The web dashboard provides:

- **Overview** — aggregate QA metrics and score trends
- **Calls** — searchable call-level evaluations and detailed score breakdowns
- **Experiments** — prompt/model comparison
- **Voice** — live Version 2 voice-agent demonstration

A call detail view exposes the underlying transcript, evaluation dimensions, primary failure, recommendation, cost and latency.

---

## Quickstart

### Requirements

- Python 3.12+
- pip
- Docker is optional

### Install

```bash
git clone https://github.com/Rollins1989/calllens.git
cd calllens

pip install -r requirements.txt
```

### Run the evaluation benchmark

```python scripts/run_eval.py```

### Run the tests

```bash
cd backend
pytest -v
cd ..
```

Expected test result:

```text
22 passed
```

### Start the application

```python -m uvicorn app.main:app --app-dir backend --reload --port 8000```

Open:

```text
http://localhost:8000
```

The dashboard is served directly by FastAPI, so there is no separate frontend build step.

### Using Make

The project also includes a `Makefile`:

```bash
make install
make dataset
make test
make eval
make run
```

### Docker

```bash
docker compose up --build
```

---

## API

FastAPI exposes interactive Swagger documentation at:

```text
http://localhost:8000/docs
```

Key endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/calls` | List calls |
| GET | `/api/calls/{call_id}` | Call + evaluation detail |
| POST | `/api/calls/{call_id}/evaluate` | Evaluate one call |
| POST | `/api/calls/evaluate-all` | Evaluate the dataset |
| GET | `/api/dashboard/summary` | Aggregate dashboard metrics |
| POST | `/api/experiments/run?group_by=` | Run prompt/model comparison |
| WS | `/ws/voice` | Live voice loop |

---

## Project Structure

```text
calllens/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── calls.py
│   │   │   ├── dashboard.py
│   │   │   ├── experiments.py
│   │   │   └── voice.py
│   │   ├── data/
│   │   │   ├── knowledge_base.json
│   │   │   ├── policies.yaml
│   │   │   ├── generate_dataset.py
│   │   │   └── synthetic_calls/
│   │   ├── evaluators/
│   │   │   ├── deterministic.py
│   │   │   ├── groundedness.py
│   │   │   ├── llm_judge.py
│   │   │   ├── policy.py
│   │   │   ├── scoring.py
│   │   │   └── sentiment.py
│   │   ├── experiments/
│   │   │   ├── runner.py
│   │   │   └── prompts/
│   │   ├── models/
│   │   ├── pipeline/
│   │   │   ├── providers.py
│   │   │   ├── voice_agent.py
│   │   │   └── cost_tracker.py
│   │   └── main.py
│   └── tests/
│       ├── test_groundedness.py
│       ├── test_intent_eval.py
│       ├── test_policy_eval.py
│       ├── test_regression.py
│       ├── test_scoring.py
│       └── test_transcript_parser.py
├── frontend/
│   └── index.html
├── scripts/
│   ├── run_eval.py
│   └── seed_db.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

## Engineering Highlights

### Explainable scoring

Each evaluation produces a weighted score across independent dimensions rather than a single unexplained LLM-generated number.

### Provider abstraction

LLM, STT and TTS providers are isolated behind interfaces so the evaluation pipeline can operate with deterministic mocks or real providers.

### Regression testing

The evaluation benchmark is designed as a regression gate. Changes to prompts, scoring logic or models can be measured against the same dataset and thresholds.

### Configuration-driven policies

Policy rules live in `policies.yaml`, while the factual reference data lives in `knowledge_base.json`. Evaluation behavior can therefore be changed without rewriting the evaluator.

### Cost and latency observability

CallLens treats model quality and operational cost/latency as related engineering metrics rather than evaluating quality in isolation.

---

## What Is Mocked?

CallLens is intentionally transparent about what runs locally without external services.

### LLM

The default LLM judge uses a deterministic mock implementation so the benchmark and tests can run without API keys, network access or inference costs.

Real providers can be configured through environment variables.

### STT / TTS

The Version 2 voice demonstration defaults to mock providers. Real provider implementations are available behind the same provider abstraction.

### Cost

Cost calculations use the project's configurable illustrative pricing table. Replace those values with the applicable provider's published pricing when using real traffic.

---

## Testing

The repository includes:

- unit tests
- integration tests
- groundedness tests
- policy evaluation tests
- scoring tests
- transcript parsing tests
- dataset-wide regression evaluation

Current local validation:

```text
22/22 backend tests passed

30-call regression:
Intent accuracy       100.0%
Resolution rate        93.3%
Hallucination rate      0.0%
Policy compliance      90.0%
Average latency      1708 ms

Regression check: PASS
```

---

## Roadmap

- PostgreSQL deployment option for multi-writer production workloads
- Persistent experiment history and metric drift tracking
- Real call recording import and transcription
- LiveKit/Pipecat integration for production voice infrastructure
- Expanded evaluation datasets
- More provider integrations
- Deployment and hosted demo

---

## Tech Stack

**Backend**
- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite

**AI / Evaluation**
- LLM-as-judge
- Prompt engineering
- Groundedness evaluation
- Policy evaluation
- Deterministic scoring
- Sentiment analysis

**Frontend**
- HTML
- CSS
- Vanilla JavaScript
- Chart.js

**Engineering**
- Pytest
- Docker
- Docker Compose
- GitHub Actions
- Make

---

## Why This Project Matters

CallLens demonstrates a different approach to conversational AI engineering:

> **Don't only build the agent. Build the system that proves whether the agent is working.**

The project combines AI evaluation, backend engineering, testing, observability, prompt experimentation and a usable developer-facing dashboard into one system.

---

## License

MIT — see [LICENSE](LICENSE).
