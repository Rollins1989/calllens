"""
Pydantic data contracts for CallLens.

These types are the backbone of the whole evaluator pipeline: a raw call
comes in as a `CallRecord`, flows through the three evaluation layers
(deterministic / LLM-judge / reference-based), and comes out as an
`EvaluationResult` that the scoring engine reduces to a single
`ScoreBreakdown`.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Speaker(str, Enum):
    agent = "agent"
    customer = "customer"


class Turn(BaseModel):
    speaker: Speaker
    text: str
    timestamp_ms: int = 0          # offset from call start
    interrupted: bool = False       # this turn cut off the other speaker


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    success: bool = True


class LatencyTrace(BaseModel):
    stt_ms: Optional[int] = None
    llm_ms: Optional[int] = None
    tts_ms: Optional[int] = None
    time_to_first_response_ms: Optional[int] = None
    end_to_end_ms: Optional[int] = None


class CostBreakdown(BaseModel):
    stt_usd: float = 0.0
    llm_input_usd: float = 0.0
    llm_output_usd: float = 0.0
    tts_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return round(
            self.stt_usd + self.llm_input_usd + self.llm_output_usd + self.tts_usd, 6
        )


class CallRecord(BaseModel):
    """A single call: either replayed from the synthetic dataset (V1) or
    produced live by the Version 2 voice pipeline."""

    model_config = ConfigDict(protected_namespaces=())

    call_id: str
    bot_type: str = "appointment_bot"
    scenario: str = "generic"
    model_tag: str = "baseline"          # which LLM/model produced the agent turns
    prompt_version: str = "v1"           # which system prompt produced the agent turns

    transcript: list[Turn]

    expected_intent: Optional[str] = None
    predicted_intent: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None

    required_actions: list[str] = Field(default_factory=list)
    actual_tool_calls: list[ToolCall] = Field(default_factory=list)

    duration_seconds: float = 0.0
    latency: LatencyTrace = Field(default_factory=LatencyTrace)
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    initial_sentiment: Optional[str] = None
    final_sentiment: Optional[str] = None


# --------------------------------------------------------------------------- #
# Evaluation outputs
# --------------------------------------------------------------------------- #

class IntentResult(BaseModel):
    expected: Optional[str]
    predicted: Optional[str]
    correct: bool


class TaskCompletionResult(BaseModel):
    required_steps: list[str]
    completed_steps: list[str]
    missing_steps: list[str]
    score: float  # 0-1


class HallucinationClaim(BaseModel):
    claim: str
    source_value: Optional[str]
    severity: str  # low | medium | high
    evidence_turn_index: Optional[int] = None


class GroundednessResult(BaseModel):
    hallucinations: list[HallucinationClaim]
    score: float  # 0-1, 1 = fully grounded


class PolicyViolation(BaseModel):
    policy_id: str
    rule: str
    severity: str
    evidence: str
    evidence_turn_index: Optional[int] = None


class PolicyResult(BaseModel):
    violations: list[PolicyViolation]
    score: float  # 0-1


class ConversationMetrics(BaseModel):
    total_turns: int
    customer_turns: int
    agent_turns: int
    avg_customer_utterance_len: float
    avg_agent_utterance_len: float
    interruption_count: int
    call_duration_seconds: float


class RubricScore(BaseModel):
    score: int  # 1-5
    reason: str


class ConversationQualityResult(BaseModel):
    relevance: RubricScore
    clarity: RubricScore
    empathy: RubricScore
    instruction_following: RubricScore
    naturalness: RubricScore
    verbosity_penalty: float = 0.0
    repetition_penalty: float = 0.0

    @property
    def average(self) -> float:
        vals = [
            self.relevance.score,
            self.clarity.score,
            self.empathy.score,
            self.instruction_following.score,
            self.naturalness.score,
        ]
        raw = (sum(vals) / len(vals)) / 5.0  # normalize 1-5 rubric to 0-1
        penalty = min(0.3, self.verbosity_penalty + self.repetition_penalty)
        return max(0.0, round(raw - penalty, 4))


class SentimentResult(BaseModel):
    initial: str
    final: str
    delta: float          # -2..+2, positive = improved
    de_escalated: bool


class ScoreBreakdown(BaseModel):
    call_id: str
    bot_type: str
    weights_used: dict[str, float]
    dimension_scores: dict[str, float]   # 0-1 per dimension
    dimension_points: dict[str, float]   # weighted contribution, 0-100 scale
    overall_score: float                 # 0-100
    primary_failure: Optional[str] = None


class EvaluationResult(BaseModel):
    call_id: str
    intent: IntentResult
    task_completion: TaskCompletionResult
    groundedness: GroundednessResult
    policy: PolicyResult
    conversation_metrics: ConversationMetrics
    conversation_quality: ConversationQualityResult
    sentiment: SentimentResult
    cost: CostBreakdown
    latency: LatencyTrace
    score: ScoreBreakdown
    recommendation: str
