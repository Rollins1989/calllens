"""Layer-1 building blocks: turn/text extraction helpers."""
from app.evaluators.deterministic import _agent_text, _customer_text
from app.models.schemas import CallRecord, Turn


def _call(turns):
    return CallRecord(
        call_id="T-1",
        transcript=[Turn(speaker=s, text=t) for s, t in turns],
    )


def test_agent_text_only_includes_agent_turns():
    call = _call([("customer", "hi"), ("agent", "hello there"), ("customer", "bye")])
    assert "hello there" in _agent_text(call)
    assert "hi" not in _agent_text(call)


def test_customer_text_only_includes_customer_turns():
    call = _call([("customer", "hi"), ("agent", "hello there")])
    assert "hi" in _customer_text(call)
    assert "hello" not in _customer_text(call)


def test_empty_transcript_returns_empty_strings():
    call = _call([])
    assert _agent_text(call) == ""
    assert _customer_text(call) == ""
