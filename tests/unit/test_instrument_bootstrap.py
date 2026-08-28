"""Unit seam: cascaid._instrument_bootstrap.bootstrap() -- the function a generated
sitecustomize.py calls in a `cascaid run`-launched child process, before the target
command's own code executes."""

from __future__ import annotations

import json

import pytest

import cascaid._instrument_bootstrap as bootstrap_module
import cascaid.ingestion.langgraph_adapter as langgraph_adapter
import cascaid.ingestion.litellm_adapter as litellm_adapter
import cascaid.ingestion.vector_query_adapter as vector_query_adapter
from cascaid.ingestion.runtime_context import current_run_id
from cascaid.ingestion.schema import NodeType


@pytest.fixture(autouse=True)
def _reset_current_run_id():
    # bootstrap() deliberately never resets current_run_id -- it's meant to live
    # for the whole process (see runtime_context.py). That's correct in
    # production but would otherwise leak across tests in this same process, so
    # force it back to the ContextVar's true default on both sides of each test
    # (not just "whatever it was before", in case an earlier test already leaked).
    current_run_id.set(None)
    yield
    current_run_id.set(None)


def test_bootstrap_sets_current_run_id_from_env(monkeypatch):
    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setenv("CASCAID_RUN_ID", "run-from-env")
    monkeypatch.delenv("CASCAID_EVENTS_PATH", raising=False)
    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrator=None, model_gateway=None, vector_db=None),
    )

    bootstrap_module.bootstrap()

    assert current_run_id.get() == "run-from-env"


def test_bootstrap_registers_langgraph_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(
        langgraph_adapter, "instrument_langgraph", lambda topology_sink: captured.update(sink=topology_sink)
    )

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrator="langgraph", model_gateway=None, vector_db=None),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured
    captured["sink"]({"planner": NodeType.AGENT}, [])
    written = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["type"] == "topology"


def test_bootstrap_registers_litellm_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(litellm_adapter, "register_litellm_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrator=None, model_gateway="litellm", vector_db=None),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_pinecone_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_pinecone_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrator=None, model_gateway=None, vector_db="pinecone"),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_weaviate_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_weaviate_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrator=None, model_gateway=None, vector_db="weaviate"),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured
