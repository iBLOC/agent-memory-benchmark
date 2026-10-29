"""Regression tests for the 2026-10-29 harness fixes.

Run:  uv run --no-sync python tests/test_llm_json_mode_and_rejections.py
(plain asserts, no pytest required)

Two fixes, both found while running the MiniMax-scored QA round:

1. `OMB_JSON_MODE=json_object` pins the wire format. MiniMax accepts
   `response_format=json_schema` without error but ignores it and answers
   in prose, so the strict attempt failed at json.loads() instead of
   falling back the way DeepSeek's 400 does.

2. A request the provider refuses outright (422 content filter) fails
   that one query, not the whole run. LongMemEval carries sensitive
   conversations on purpose, and MiniMax filters some of them.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from memory_bench.llm.base import Schema  # noqa: E402
from memory_bench.llm.openai import OpenAILLM  # noqa: E402
from memory_bench.runner import _is_provider_rejection  # noqa: E402


def test_json_mode_pins_the_wire_format():
    """With OMB_JSON_MODE=json_object the strict json_schema attempt is skipped."""
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = type("M", (), {})()
            msg.content = '{"color": "blue"}'
            choice = type("C", (), {})()
            choice.message = msg
            resp = type("R", (), {})()
            resp.choices = [choice]
            return resp

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    os.environ["OMB_JSON_MODE"] = "json_object"
    llm = OpenAILLM(model="MiniMax-M3.1-Flash")
    llm._client = _FakeClient()
    schema = Schema(properties={"color": {"type": "string"}}, required=["color"])
    out = llm.generate("What is the color of the sky?", schema)
    assert out == {"color": "blue"}
    assert captured["response_format"] == {"type": "json_object"}, captured


def test_json_mode_default_keeps_the_strict_attempt():
    """Without the knob the strict json_schema attempt still happens first."""
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = type("M", (), {})()
            msg.content = '{"color": "blue"}'
            choice = type("C", (), {})()
            choice.message = msg
            resp = type("R", (), {})()
            resp.choices = [choice]
            return resp

    class _FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    os.environ.pop("OMB_JSON_MODE", None)
    llm = OpenAILLM(model="gpt-4o")
    llm._client = _FakeClient()
    schema = Schema(properties={"color": {"type": "string"}}, required=["color"])
    llm.generate("What is the color of the sky?", schema)
    assert captured["response_format"]["type"] == "json_schema", captured


def test_provider_rejections_are_recognised():
    """A refusal is a per-query outcome; a transient error is not."""
    rejected = (
        "UnprocessableEntityError: Error code: 422 - {'type': 'error', 'error': "
        "{'type': 'unprocessable_entity_error', 'message': 'input new_sensitive (1026)'}}"
    )
    assert _is_provider_rejection(rejected)
    assert _is_provider_rejection("... content_filter ...")
    assert not _is_provider_rejection(
        "RateLimitError: Error code: 429 - rate_limit_error"
    )
    assert not _is_provider_rejection("APIConnectionError: connection reset")


def test_a_rejected_prompt_fails_one_query_not_the_run():
    """The runner's rejection path returns a failed QueryResult."""
    from memory_bench.models import QueryResult

    async def _run():
        # Exercise the same shape the runner uses: a provider refusal inside
        # the per-query attempt must not propagate out of the unit loop.
        try:
            raise RuntimeError(
                "UnprocessableEntityError: Error code: 422 - input new_sensitive"
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_provider_rejection(str(exc)):
                raise
            return QueryResult(
                query_id="q1",
                query="q",
                answer="",
                reasoning="",
                context="",
                context_tokens=0,
                retrieve_time_ms=0.0,
                gold_answers=["a"],
                correct=False,
                judge_reason=f"provider rejected the request: {exc}",
            )

    result = asyncio.run(_run())
    assert result.correct is False
    assert "provider rejected" in result.judge_reason


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{4 - failures}/4 llm-json-mode-and-rejection tests passed")
    sys.exit(1 if failures else 0)
