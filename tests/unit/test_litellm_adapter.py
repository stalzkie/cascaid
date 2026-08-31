import asyncio
import time
from datetime import datetime, timezone

import litellm
import openai

from cascaid.ingestion.litellm_adapter import (
    _build_async_cascaid_logger,
    _snapshot_context_into_metadata,
    inside_litellm_dispatch,
    litellm_failure_to_call_event,
    litellm_success_to_call_event,
    register_litellm_callbacks,
)
from cascaid.ingestion.runtime_context import current_run_id, track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType


def _wait_for(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("timed out waiting for litellm's async success callback to fire")


def test_converts_real_litellm_success_callback_into_call_event():
    captured = {}

    # The adapter must be invoked *inside* litellm's own callback dispatch, since
    # current_node is only readable within the context litellm copies at call time --
    # reading it later from unrelated code (e.g. the test body) sees the default.
    def callback(kwargs, completion_response, start_time, end_time):
        captured["event"] = litellm_success_to_call_event(
            kwargs, completion_response, start_time, end_time, run_id="run-1", step=0
        )

    litellm.success_callback = [callback]
    try:
        with track_node("research_agent"):
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello there",
            )
        # exits the `with` block (and resets the contextvar) before the callback has
        # necessarily fired -- litellm dispatches it asynchronously, so this also proves
        # the caller attribution survives that race, not just that it works when lucky.
        _wait_for(lambda: "event" in captured)
    finally:
        litellm.success_callback = []

    event = captured["event"]

    assert event.run_id == "run-1"
    assert event.scenario == "production"
    assert event.step == 0
    assert event.caller == "research_agent"
    assert event.caller_type == NodeType.AGENT
    assert event.callee == "gpt-4o-mini"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is False
    assert event.retried is False
    assert event.latency_ms >= 0
    assert event.token_cost >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_converts_real_litellm_failure_callback_into_call_event():
    captured = {}

    def callback(kwargs, completion_response, start_time, end_time):
        captured["event"] = litellm_failure_to_call_event(
            kwargs, completion_response, start_time, end_time, run_id="run-1", step=3
        )

    litellm.failure_callback = [callback]
    try:
        with track_node("research_agent"):
            try:
                litellm.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    mock_response=Exception("rate limit exceeded"),
                )
            except Exception:
                pass
        _wait_for(lambda: "event" in captured)
    finally:
        litellm.failure_callback = []

    event = captured["event"]

    assert event.run_id == "run-1"
    assert event.step == 3
    assert event.caller == "research_agent"
    assert event.callee == "gpt-4o-mini"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is True
    assert event.retried is False
    assert event.latency_ms >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_register_litellm_callbacks_appends_without_clobbering_existing_callbacks():
    existing_success = lambda *a: None  # noqa: E731 -- stand-in for e.g. Langfuse's existing callback
    existing_failure = lambda *a: None  # noqa: E731

    class _ExistingLogger:  # stand-in for e.g. Langfuse's existing CustomLogger
        pass

    existing_logger = _ExistingLogger()
    litellm.success_callback = [existing_success]
    litellm.failure_callback = [existing_failure]
    litellm.callbacks = [existing_logger]
    try:
        register_litellm_callbacks(sink=lambda event: None)

        assert existing_success in litellm.success_callback
        assert len(litellm.success_callback) == 2
        assert existing_failure in litellm.failure_callback
        assert len(litellm.failure_callback) == 2
        assert existing_logger in litellm.callbacks
        assert len(litellm.callbacks) == 2
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []


def test_registered_callback_sinks_a_call_event_when_run_context_is_set():
    captured = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        register_litellm_callbacks(sink=captured.append)

        with track_run("run-1"), track_step(2), track_node("research_agent"):
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello there",
            )
        _wait_for(lambda: len(captured) == 1)
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.callee == "gpt-4o-mini"
    assert event.error is False


def test_registered_callback_sinks_exactly_one_event_for_a_streaming_sync_call():
    # Regression test: a streaming sync completion() call is dispatched by
    # litellm via a background ThreadPoolExecutor (verified empirically --
    # see docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md),
    # which does not propagate contextvars, AND fires one success callback per
    # chunk plus a final aggregated one. Without both fixes (metadata-based
    # attribution + skipping non-final chunks), streaming either drops every
    # event or produces a run of near-duplicate, degenerate-latency ones.
    captured = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        register_litellm_callbacks(sink=captured.append)

        with track_run("run-1"), track_step(2), track_node("research_agent"):
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                mock_response="hello there streaming",
                stream=True,
            )
            for _ in response:
                pass
        _wait_for(lambda: len(captured) == 1, timeout=5.0, interval=0.05)
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.callee == "gpt-4o-mini"
    assert event.error is False


def test_registered_callback_skips_the_sink_when_run_context_is_not_set():
    captured = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        register_litellm_callbacks(sink=captured.append)

        # No track_run/track_step block -- instrumentation hasn't reached this call.
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello there",
        )
        time.sleep(0.3)  # give the async callback a chance to fire, then prove it didn't sink anything
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert captured == []


def test_inside_litellm_dispatch_is_true_only_around_the_real_dispatch():
    # Regression test for ADR 0001's OpenAI dedup mechanism (openai_adapter.py):
    # litellm's OpenAI provider path calls
    # openai_client.chat.completions.with_raw_response.create(...) internally, which
    # resolves through the same Completions.create the openai adapter patches
    # (verified empirically -- with_raw_response is a @cached_property that looks up
    # completions.create dynamically). This flag is how the openai adapter tells that
    # call apart from a customer's own direct create() call.
    captured = []
    original_create = openai.resources.chat.completions.Completions.create

    def probe_create(self, *args, **kwargs):
        captured.append(inside_litellm_dispatch.get())
        raise RuntimeError("stand-in for a real HTTP call -- only the flag matters here")

    openai.resources.chat.completions.Completions.create = probe_create
    try:
        register_litellm_callbacks(sink=lambda event: None)  # ensures completion() is patched
        assert inside_litellm_dispatch.get() is False

        try:
            # litellm caches OpenAI client instances by params (get_cached_openai_client) --
            # a with_raw_response accessed on a client cached from an earlier test would
            # keep pointing at *that* test's Completions.create, not this probe. A unique
            # api_key forces a fresh, uncached client so this test sees its own patch.
            litellm.completion(
                model="gpt-4o-mini",
                api_key="test-key-dispatch-flag-probe",
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception:
            pass  # probe_create's RuntimeError propagates through litellm -- expected
    finally:
        openai.resources.chat.completions.Completions.create = original_create
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert captured == [True]
    assert inside_litellm_dispatch.get() is False


def test_snapshot_context_into_metadata_captures_run_step_node():
    with track_run("run-1"), track_step(2), track_node("research_agent"):
        kwargs = _snapshot_context_into_metadata({"model": "gpt-4o-mini"})

    assert kwargs["metadata"] == {
        "cascaid_run_id": "run-1",
        "cascaid_step": 2,
        "cascaid_node": "research_agent",
    }


def test_snapshot_context_into_metadata_merges_with_existing_customer_metadata():
    with track_run("run-1"), track_step(2), track_node("research_agent"):
        kwargs = _snapshot_context_into_metadata({"model": "gpt-4o-mini", "metadata": {"trace_id": "abc"}})

    assert kwargs["metadata"]["trace_id"] == "abc"
    assert kwargs["metadata"]["cascaid_run_id"] == "run-1"


def test_snapshot_context_into_metadata_leaves_kwargs_untouched_without_run_context():
    kwargs = _snapshot_context_into_metadata({"model": "gpt-4o-mini"})

    assert "metadata" not in kwargs


def test_async_log_success_event_reads_the_metadata_snapshot_not_ambient_context():
    # litellm defers async logging to run detached from the coroutine that made
    # the call -- current_run_id/current_step/current_node are already reset
    # (None) by the time this fires for real (verified empirically, see
    # docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md).
    # Proves attribution comes from the metadata snapshot captured at call
    # time, not a (by-then-stale) contextvar read.
    captured = []
    logger = _build_async_cascaid_logger(sink=captured.append)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {"cascaid_run_id": "run-1", "cascaid_step": 2, "cascaid_node": "research_agent"}
        },
    }
    now = datetime.now(timezone.utc)

    assert current_run_id.get() is None  # simulates the context already being gone
    asyncio.run(logger.async_log_success_event(kwargs, None, now, now))

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "research_agent"
    assert event.callee == "gpt-4o-mini"


def test_async_log_failure_event_reads_the_metadata_snapshot():
    captured = []
    logger = _build_async_cascaid_logger(sink=captured.append)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {"cascaid_run_id": "run-1", "cascaid_step": 2, "cascaid_node": "research_agent"}
        },
    }
    now = datetime.now(timezone.utc)

    asyncio.run(logger.async_log_failure_event(kwargs, None, now, now))

    assert captured[0].error is True
    assert captured[0].callee == "gpt-4o-mini"


def test_async_log_success_event_skips_the_sink_without_a_metadata_snapshot():
    # e.g. a litellm call the customer made outside any tracked run -- no
    # metadata was ever injected, so there's nothing to attribute this to.
    captured = []
    logger = _build_async_cascaid_logger(sink=captured.append)
    now = datetime.now(timezone.utc)

    asyncio.run(logger.async_log_success_event({"model": "gpt-4o-mini"}, None, now, now))

    assert captured == []
