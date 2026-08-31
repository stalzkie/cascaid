import asyncio
from datetime import timezone

import pytest

from cascaid.ingestion.manual_adapter import _default_sink, observe_call, observe_call_async
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType


def test_observe_call_records_success_and_sinks_it():
    captured = []
    with track_run("run-1"), track_step(2), track_node("my_agent"):
        with observe_call("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append) as call:
            pass

    assert call.event is not None
    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "my_agent"
    assert event.caller_type == NodeType.AGENT
    assert event.callee == "my_model"
    assert event.callee_type == NodeType.MODEL_ENDPOINT
    assert event.error is False
    assert event.retried is False
    assert event.token_cost == 0.0
    assert event.latency_ms >= 0
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_observe_call_uses_explicit_caller_and_caller_type_when_given():
    captured = []
    with track_run("run-1"), track_step(0):
        with observe_call(
            "my_model",
            NodeType.MODEL_ENDPOINT,
            caller="explicit_caller",
            caller_type=NodeType.TOOL,
            sink=captured.append,
        ):
            pass

    assert captured[0].caller == "explicit_caller"
    assert captured[0].caller_type == NodeType.TOOL


def test_observe_call_records_error_and_reraises():
    captured = []
    with track_run("run-1"), track_step(3), track_node("my_agent"):
        with pytest.raises(RuntimeError, match="boom"):
            with observe_call("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append):
                raise RuntimeError("boom")

    assert len(captured) == 1
    assert captured[0].error is True


def test_observe_call_skips_the_sink_when_run_context_is_not_set():
    captured = []
    # No track_run/track_step block -- nothing has established run context yet
    # (the exact case for hand-rolled orchestration with no auto-detected adapter).
    with observe_call("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append) as call:
        pass

    assert captured == []
    assert call.event is None


def test_observe_call_skips_the_sink_when_only_run_id_is_set_not_step():
    captured = []
    with track_run("run-1"):
        with observe_call("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append):
            pass

    assert captured == []


def test_observe_call_defaults_to_default_sink_when_none_given():
    # Proves the default-sink wiring exists without depending on real file I/O --
    # covered separately by test_default_sink_writes_call_events_path below.
    with track_run("run-1"), track_step(0):
        with observe_call("my_model", NodeType.MODEL_ENDPOINT) as call:
            pass

    assert call.event is not None  # built regardless of where _default_sink sends it


def test_default_sink_writes_a_json_line_to_cascaid_events_path(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    with track_run("run-1"), track_step(1), track_node("my_agent"):
        with observe_call("my_model", NodeType.MODEL_ENDPOINT):
            pass

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"type": "call_event"' in lines[0]
    assert '"callee": "my_model"' in lines[0]


def test_default_sink_no_ops_when_cascaid_events_path_is_unset(monkeypatch):
    monkeypatch.delenv("CASCAID_EVENTS_PATH", raising=False)
    from cascaid.ingestion.schema import CallEvent

    event = CallEvent(
        run_id="run-1",
        scenario="production",
        step=0,
        caller="a",
        callee="b",
        caller_type=NodeType.AGENT,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=1.0,
        error=False,
        retried=False,
        token_cost=0.0,
    )
    _default_sink(event)  # must not raise


def test_observe_call_async_records_success_and_sinks_it():
    captured = []

    async def run():
        with track_run("run-1"), track_step(4), track_node("my_agent"):
            async with observe_call_async("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append):
                pass

    asyncio.run(run())

    assert len(captured) == 1
    event = captured[0]
    assert event.run_id == "run-1"
    assert event.step == 4
    assert event.caller == "my_agent"
    assert event.callee == "my_model"
    assert event.error is False


def test_observe_call_async_records_error_and_reraises():
    captured = []

    async def run():
        with track_run("run-1"), track_step(5), track_node("my_agent"):
            with pytest.raises(RuntimeError, match="boom"):
                async with observe_call_async("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append):
                    raise RuntimeError("boom")

    asyncio.run(run())

    assert len(captured) == 1
    assert captured[0].error is True


def test_observe_call_async_skips_the_sink_when_run_context_is_not_set():
    captured = []

    async def run():
        async with observe_call_async("my_model", NodeType.MODEL_ENDPOINT, sink=captured.append):
            pass

    asyncio.run(run())

    assert captured == []
