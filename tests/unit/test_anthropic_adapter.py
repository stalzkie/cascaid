from datetime import timezone

import anthropic
import httpx
from anthropic.types import Message, TextBlock, Usage

from cascaid.ingestion.anthropic_adapter import instrument_anthropic
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType


def _real_message(model="claude-opus-4-6") -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(text="hello there", type="text")],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def test_converts_a_real_sync_anthropic_call_into_a_call_event():
    captured = []
    original_create = anthropic.resources.messages.Messages.create

    def fake_create(self, *args, **kwargs):
        return _real_message(model=kwargs["model"])

    anthropic.resources.messages.Messages.create = fake_create
    try:
        instrument_anthropic(sink=captured.append)
        client = anthropic.Anthropic(api_key="test-key")

        with track_run("run-1"), track_step(2), track_node("research_agent"):
            client.messages.create(
                model="claude-opus-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        anthropic.resources.messages.Messages.create = original_create

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.caller_type == NodeType.AGENT
    assert event.callee == "claude-opus-4-6"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is False
    assert event.retried is False
    assert event.latency_ms >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_sinks_an_error_call_event_and_reraises_when_the_sdk_call_fails():
    captured = []
    original_create = anthropic.resources.messages.Messages.create

    def failing_create(self, *args, **kwargs):
        resp = httpx.Response(500, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        raise anthropic.APIStatusError("boom", response=resp, body=None)

    anthropic.resources.messages.Messages.create = failing_create
    try:
        instrument_anthropic(sink=captured.append)
        client = anthropic.Anthropic(api_key="test-key")

        raised = None
        with track_run("run-1"), track_step(3), track_node("research_agent"):
            try:
                client.messages.create(
                    model="claude-opus-4-6", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
                )
            except anthropic.APIStatusError as exc:
                raised = exc
    finally:
        anthropic.resources.messages.Messages.create = original_create

    assert raised is not None  # the customer's own exception handling must still see it
    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 3
    assert event.callee == "claude-opus-4-6"
    assert event.error is True


def test_skips_the_sink_when_run_context_is_not_set():
    captured = []
    original_create = anthropic.resources.messages.Messages.create

    def fake_create(self, *args, **kwargs):
        return _real_message(model=kwargs["model"])

    anthropic.resources.messages.Messages.create = fake_create
    try:
        instrument_anthropic(sink=captured.append)
        client = anthropic.Anthropic(api_key="test-key")

        # No track_run/track_step block -- instrumentation hasn't reached this call.
        client.messages.create(model="claude-opus-4-6", max_tokens=100, messages=[{"role": "user", "content": "hi"}])
    finally:
        anthropic.resources.messages.Messages.create = original_create

    assert captured == []


def test_converts_a_real_async_anthropic_call_into_a_call_event():
    import asyncio

    captured = []
    original_create = anthropic.resources.messages.AsyncMessages.create

    async def fake_create(self, *args, **kwargs):
        return _real_message(model=kwargs["model"])

    anthropic.resources.messages.AsyncMessages.create = fake_create
    try:
        instrument_anthropic(sink=captured.append)
        client = anthropic.AsyncAnthropic(api_key="test-key")

        async def run():
            with track_run("run-1"), track_step(4), track_node("research_agent"):
                await client.messages.create(
                    model="claude-opus-4-6", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
                )

        asyncio.run(run())
    finally:
        anthropic.resources.messages.AsyncMessages.create = original_create

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 4
    assert event.caller == "research_agent"
    assert event.callee == "claude-opus-4-6"
    assert event.error is False


def test_instrument_anthropic_does_not_stack_duplicate_instrumentation_on_repeated_calls():
    captured = []
    original_create = anthropic.resources.messages.Messages.create

    def fake_create(self, *args, **kwargs):
        return _real_message(model=kwargs["model"])

    anthropic.resources.messages.Messages.create = fake_create
    try:
        instrument_anthropic(sink=captured.append)
        instrument_anthropic(sink=captured.append)  # simulates a repeated bootstrap call
        client = anthropic.Anthropic(api_key="test-key")

        with track_run("run-1"), track_step(0), track_node("research_agent"):
            client.messages.create(
                model="claude-opus-4-6", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        anthropic.resources.messages.Messages.create = original_create

    assert len(captured) == 1
