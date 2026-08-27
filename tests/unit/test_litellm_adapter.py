import time

import litellm

from cascaid.ingestion.litellm_adapter import litellm_failure_to_call_event, litellm_success_to_call_event
from cascaid.ingestion.runtime_context import track_node
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
