from datetime import timezone

import google.genai as genai
from google.genai import errors, types

from cascaid.ingestion.gemini_adapter import instrument_gemini
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType


def _real_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="hello there")], role="model"),
                finish_reason="STOP",
            )
        ]
    )


def test_converts_a_real_sync_gemini_call_into_a_call_event():
    captured = []
    original_create = genai.models.Models.generate_content

    def fake_generate_content(self, *args, **kwargs):
        return _real_response()

    genai.models.Models.generate_content = fake_generate_content
    try:
        instrument_gemini(sink=captured.append)
        client = genai.Client(api_key="test-key")

        with track_run("run-1"), track_step(2), track_node("research_agent"):
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    finally:
        genai.models.Models.generate_content = original_create

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.caller_type == NodeType.AGENT
    assert event.callee == "gemini-2.5-flash"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is False
    assert event.retried is False
    assert event.latency_ms >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_sinks_an_error_call_event_and_reraises_when_the_sdk_call_fails():
    captured = []
    original_create = genai.models.Models.generate_content

    def failing_generate_content(self, *args, **kwargs):
        raise errors.ClientError(code=500, response_json={})

    genai.models.Models.generate_content = failing_generate_content
    try:
        instrument_gemini(sink=captured.append)
        client = genai.Client(api_key="test-key")

        raised = None
        with track_run("run-1"), track_step(3), track_node("research_agent"):
            try:
                client.models.generate_content(model="gemini-2.5-flash", contents="hi")
            except errors.ClientError as exc:
                raised = exc
    finally:
        genai.models.Models.generate_content = original_create

    assert raised is not None
    assert len(captured) == 1
    assert captured[0].callee == "gemini-2.5-flash"
    assert captured[0].error is True


def test_skips_the_sink_when_run_context_is_not_set():
    captured = []
    original_create = genai.models.Models.generate_content

    def fake_generate_content(self, *args, **kwargs):
        return _real_response()

    genai.models.Models.generate_content = fake_generate_content
    try:
        instrument_gemini(sink=captured.append)
        client = genai.Client(api_key="test-key")

        client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    finally:
        genai.models.Models.generate_content = original_create

    assert captured == []


def test_converts_a_real_async_gemini_call_into_a_call_event():
    import asyncio

    captured = []
    original_create = genai.models.AsyncModels.generate_content

    async def fake_generate_content(self, *args, **kwargs):
        return _real_response()

    genai.models.AsyncModels.generate_content = fake_generate_content
    try:
        instrument_gemini(sink=captured.append)
        client = genai.Client(api_key="test-key")

        async def run():
            with track_run("run-1"), track_step(4), track_node("research_agent"):
                await client.aio.models.generate_content(model="gemini-2.5-flash", contents="hi")

        asyncio.run(run())
    finally:
        genai.models.AsyncModels.generate_content = original_create

    assert len(captured) == 1
    assert captured[0].callee == "gemini-2.5-flash"


def test_instrument_gemini_does_not_stack_duplicate_instrumentation_on_repeated_calls():
    captured = []
    original_create = genai.models.Models.generate_content

    def fake_generate_content(self, *args, **kwargs):
        return _real_response()

    genai.models.Models.generate_content = fake_generate_content
    try:
        instrument_gemini(sink=captured.append)
        instrument_gemini(sink=captured.append)  # simulates a repeated bootstrap call
        client = genai.Client(api_key="test-key")

        with track_run("run-1"), track_step(0), track_node("research_agent"):
            client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    finally:
        genai.models.Models.generate_content = original_create

    assert len(captured) == 1
