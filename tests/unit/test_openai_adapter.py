from datetime import timezone

import litellm
import openai
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from cascaid.ingestion.litellm_adapter import register_litellm_callbacks
from cascaid.ingestion.openai_adapter import instrument_openai
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType


def _real_chat_completion(model="gpt-4o-mini") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )


class _FakeRawResponse:
    """Stand-in for openai's LegacyAPIResponse -- litellm's provider path calls
    .with_raw_response.create(...), which needs .headers and .parse() on whatever
    Completions.create resolves to at the time with_raw_response is first accessed."""

    def __init__(self, completion: ChatCompletion):
        self.headers = {}
        self._completion = completion

    def parse(self) -> ChatCompletion:
        return self._completion


def test_converts_a_real_direct_sync_openai_call_into_a_call_event():
    captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def fake_create(self, *args, **kwargs):
        return _real_chat_completion(model=kwargs["model"])

    openai.resources.chat.completions.Completions.create = fake_create
    try:
        instrument_openai(sink=captured.append)
        client = openai.OpenAI(api_key="test-key")

        with track_run("run-1"), track_step(2), track_node("research_agent"):
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    finally:
        openai.resources.chat.completions.Completions.create = original_create

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.caller_type == NodeType.AGENT
    assert event.callee == "gpt-4o-mini"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is False
    assert event.retried is False
    assert event.latency_ms >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_sinks_an_error_call_event_and_reraises_when_the_direct_sdk_call_fails():
    captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def failing_create(self, *args, **kwargs):
        raise openai.APIConnectionError(request=None)

    openai.resources.chat.completions.Completions.create = failing_create
    try:
        instrument_openai(sink=captured.append)
        client = openai.OpenAI(api_key="test-key")

        raised = None
        with track_run("run-1"), track_step(3), track_node("research_agent"):
            try:
                client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
            except openai.APIConnectionError as exc:
                raised = exc
    finally:
        openai.resources.chat.completions.Completions.create = original_create

    assert raised is not None
    assert len(captured) == 1
    assert captured[0].error is True
    assert captured[0].callee == "gpt-4o-mini"


def test_skips_the_sink_when_run_context_is_not_set():
    captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def fake_create(self, *args, **kwargs):
        return _real_chat_completion(model=kwargs["model"])

    openai.resources.chat.completions.Completions.create = fake_create
    try:
        instrument_openai(sink=captured.append)
        client = openai.OpenAI(api_key="test-key")

        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    finally:
        openai.resources.chat.completions.Completions.create = original_create

    assert captured == []


def test_converts_a_real_direct_async_openai_call_into_a_call_event():
    import asyncio

    captured = []
    original_create = openai.resources.chat.completions.AsyncCompletions.create

    async def fake_create(self, *args, **kwargs):
        return _real_chat_completion(model=kwargs["model"])

    openai.resources.chat.completions.AsyncCompletions.create = fake_create
    try:
        instrument_openai(sink=captured.append)
        client = openai.AsyncOpenAI(api_key="test-key")

        async def run():
            with track_run("run-1"), track_step(4), track_node("research_agent"):
                await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

        asyncio.run(run())
    finally:
        openai.resources.chat.completions.AsyncCompletions.create = original_create

    assert len(captured) == 1
    assert captured[0].callee == "gpt-4o-mini"


def test_instrument_openai_does_not_stack_duplicate_instrumentation_on_repeated_calls():
    captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def fake_create(self, *args, **kwargs):
        return _real_chat_completion(model=kwargs["model"])

    openai.resources.chat.completions.Completions.create = fake_create
    try:
        instrument_openai(sink=captured.append)
        instrument_openai(sink=captured.append)  # simulates a repeated bootstrap call
        client = openai.OpenAI(api_key="test-key")

        with track_run("run-1"), track_step(0), track_node("research_agent"):
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    finally:
        openai.resources.chat.completions.Completions.create = original_create

    assert len(captured) == 1


def test_a_litellm_dispatched_openai_call_is_sinked_once_by_litellm_not_twice_by_openai_adapter():
    # The composability contract from ADR 0001: both adapters active at once, a
    # litellm.completion() call for an OpenAI model must produce exactly one CallEvent
    # (from litellm_adapter's own callback registry), not two.
    openai_captured = []
    litellm_captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def fake_create(self, *args, **kwargs):
        return _FakeRawResponse(_real_chat_completion(model=kwargs["model"]))

    openai.resources.chat.completions.Completions.create = fake_create
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        instrument_openai(sink=openai_captured.append)
        register_litellm_callbacks(sink=litellm_captured.append)

        # A unique api_key forces litellm to construct a fresh, uncached OpenAI client
        # (get_cached_openai_client caches by params) so this test's Completions.create
        # patch is the one with_raw_response actually resolves, not a stale one cached
        # from another test that also called litellm.completion for an OpenAI model.
        with track_run("run-1"), track_step(1), track_node("research_agent"):
            litellm.completion(
                model="gpt-4o-mini", api_key="test-key-dedup-scenario", messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        openai.resources.chat.completions.Completions.create = original_create
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert openai_captured == []
    assert len(litellm_captured) == 1
    assert litellm_captured[0].callee == "gpt-4o-mini"


def test_a_direct_openai_call_still_sinks_while_litellm_instrumentation_is_also_active():
    # Proves inside_litellm_dispatch scopes only to litellm's own dispatch window, not
    # globally whenever litellm_adapter has been registered -- a customer's own direct
    # openai.OpenAI().chat.completions.create() call must still be observed.
    openai_captured = []
    litellm_captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def fake_create(self, *args, **kwargs):
        return _real_chat_completion(model=kwargs["model"])

    openai.resources.chat.completions.Completions.create = fake_create
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        instrument_openai(sink=openai_captured.append)
        register_litellm_callbacks(sink=litellm_captured.append)
        client = openai.OpenAI(api_key="test-key")

        with track_run("run-1"), track_step(1), track_node("research_agent"):
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    finally:
        openai.resources.chat.completions.Completions.create = original_create
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert len(openai_captured) == 1
    assert litellm_captured == []
