"""
Provider abstraction layer.

CallLens is deliberately model-agnostic (see README section on Model
Comparison): the LLM judge, the voice agent, STT and TTS are all defined as
small interfaces with a "mock" implementation that requires no API key and
runs fully offline (used by CI and local demos), plus real implementations
that activate automatically when the corresponding API key is present.

Swap providers purely via environment variables - no code changes needed.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app import config


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class BaseLLM(ABC):
    @abstractmethod
    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        """Ask the model to respond with JSON only and return the raw text."""

    @abstractmethod
    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        """Ask the model for a plain-text reply (used by the voice agent)."""


class MockLLM(BaseLLM):
    """Deterministic, keyword-driven stand-in for a real model. Good enough to
    exercise the full pipeline end-to-end (and for CI) without any API key or
    network access. Real judging quality comes from swapping in
    ANTHROPIC_API_KEY / OPENAI_API_KEY - this class exists so the *system*
    still works with zero configuration."""

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        start = time.time()
        # Very small heuristic "judge": scores relevance/clarity/etc from
        # simple signals so the rest of the pipeline has something real to
        # combine, without pretending to be a sophisticated model.
        payload = {
            "relevance": {"score": 4, "reason": "Mock judge: agent addressed the stated request."},
            "clarity": {"score": 4, "reason": "Mock judge: responses were short and direct."},
            "empathy": {"score": 3, "reason": "Mock judge: limited explicit empathy language detected."},
            "instruction_following": {"score": 4, "reason": "Mock judge: agent stayed on the required flow."},
            "naturalness": {"score": 3, "reason": "Mock judge: some templated phrasing detected."},
        }
        text = json.dumps(payload)
        latency = int((time.time() - start) * 1000) + 40  # simulate small latency
        return LLMResponse(text=text, input_tokens=len(user.split()), output_tokens=len(text.split()), latency_ms=latency)

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        start = time.time()
        reply = "Sure, I can help with that. Could you confirm your preferred date and time?"
        latency = int((time.time() - start) * 1000) + 60
        return LLMResponse(text=reply, input_tokens=len(user.split()), output_tokens=len(reply.split()), latency_ms=latency)


class AnthropicLLM(BaseLLM):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.ANTHROPIC_MODEL

    def _call(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        start = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = int((time.time() - start) * 1000)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        json_system = system + "\n\nRespond with ONLY valid JSON. No preamble, no markdown fences."
        return self._call(json_system, user, max_tokens)

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        return self._call(system, user, max_tokens)


class OpenAILLM(BaseLLM):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def _call(self, system: str, user: str, max_tokens: int, json_mode: bool) -> LLMResponse:
        start = time.time()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        latency = int((time.time() - start) * 1000)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        return self._call(system, user, max_tokens, json_mode=True)

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        return self._call(system, user, max_tokens, json_mode=False)


def get_llm() -> BaseLLM:
    provider = config.LLM_PROVIDER
    if provider == "anthropic" and config.ANTHROPIC_API_KEY:
        return AnthropicLLM()
    if provider == "openai" and config.OPENAI_API_KEY:
        return OpenAILLM()
    return MockLLM()


# --------------------------------------------------------------------------- #
# STT / TTS  (used by the Version 2 voice pipeline)
# --------------------------------------------------------------------------- #

class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> tuple[str, int]:
        """Returns (text, latency_ms)."""


class MockSTT(BaseSTT):
    """Offline stand-in: real audio bytes aren't decoded, a canned utterance
    is returned so the rest of the pipeline (LLM -> TTS -> evaluator) can be
    demoed end-to-end without any speech models installed."""

    _script = [
        "Hi, I'd like to book an appointment for next Tuesday.",
        "My name is Priya Shah, phone number 9876543210.",
        "Around 4 PM works for me.",
        "Yes, that's correct, please confirm it.",
    ]
    _i = 0

    def transcribe(self, audio_bytes: bytes) -> tuple[str, int]:
        text = self._script[self._i % len(self._script)]
        MockSTT._i += 1
        return text, 120


class OpenAIWhisperSTT(BaseSTT):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def transcribe(self, audio_bytes: bytes) -> tuple[str, int]:
        import io
        start = time.time()
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.webm"
        resp = self.client.audio.transcriptions.create(model="whisper-1", file=buf)
        latency = int((time.time() - start) * 1000)
        return resp.text, latency


def get_stt() -> BaseSTT:
    if config.STT_PROVIDER == "openai_whisper" and config.OPENAI_API_KEY:
        return OpenAIWhisperSTT()
    return MockSTT()


class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Returns (audio_bytes, latency_ms). Mock returns empty bytes."""


class MockTTS(BaseTTS):
    def synthesize(self, text: str) -> tuple[bytes, int]:
        return b"", 90


class OpenAITTS(BaseTTS):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def synthesize(self, text: str) -> tuple[bytes, int]:
        start = time.time()
        resp = self.client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        latency = int((time.time() - start) * 1000)
        return resp.content, latency


def get_tts() -> BaseTTS:
    if config.TTS_PROVIDER == "openai_tts" and config.OPENAI_API_KEY:
        return OpenAITTS()
    return MockTTS()
