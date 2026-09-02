"""Unit seam: cascaid._instrument_bootstrap.bootstrap() -- the function a generated
sitecustomize.py calls in a `cascaid run`-launched child process, before the target
command's own code executes."""

from __future__ import annotations

import json

import pytest

import cascaid._instrument_bootstrap as bootstrap_module
import cascaid.ingestion.anthropic_adapter as anthropic_adapter
import cascaid.ingestion.autogen_adapter as autogen_adapter
import cascaid.ingestion.crewai_adapter as crewai_adapter
import cascaid.ingestion.gemini_adapter as gemini_adapter
import cascaid.ingestion.langgraph_adapter as langgraph_adapter
import cascaid.ingestion.litellm_adapter as litellm_adapter
import cascaid.ingestion.openai_adapter as openai_adapter
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
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset()),
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
        lambda: DetectedStack(orchestrators=frozenset({"langgraph"}), model_gateway=None, vector_dbs=frozenset()),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured
    captured["sink"]({"planner": NodeType.AGENT}, [])
    written = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["type"] == "topology"


def test_bootstrap_registers_crewai_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(crewai_adapter, "instrument_crewai", lambda topology_sink: captured.update(sink=topology_sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset({"crewai"}), model_gateway=None, vector_dbs=frozenset()),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured
    captured["sink"]({"researcher (0)": NodeType.AGENT}, [])
    written = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["type"] == "topology"


def test_bootstrap_registers_autogen_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(
        autogen_adapter, "instrument_autogen", lambda topology_sink: captured.update(sink=topology_sink)
    )

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset({"autogen"}), model_gateway=None, vector_dbs=frozenset()),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured
    captured["sink"]({"agent_a": NodeType.AGENT}, [])
    written = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["type"] == "topology"


def test_bootstrap_registers_both_orchestrators_when_both_are_available(monkeypatch, tmp_path):
    # Regression test: cascaid's own hard dependency on langgraph (for `cascaid
    # demo`) means langgraph is importable in every real install, so orchestrator
    # detection must not be exclusive -- a customer whose app only uses CrewAI
    # still needs it instrumented even though langgraph is also present.
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(
        langgraph_adapter, "instrument_langgraph", lambda topology_sink: captured.update(langgraph=topology_sink)
    )
    monkeypatch.setattr(
        crewai_adapter, "instrument_crewai", lambda topology_sink: captured.update(crewai=topology_sink)
    )

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset({"langgraph", "crewai"}), model_gateway=None, vector_dbs=frozenset()
        ),
    )

    bootstrap_module.bootstrap()

    assert "langgraph" in captured
    assert "crewai" in captured


def test_bootstrap_registers_litellm_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(litellm_adapter, "register_litellm_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway="litellm", vector_dbs=frozenset()),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_anthropic_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(anthropic_adapter, "instrument_anthropic", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset(), direct_sdks=frozenset({"anthropic"})
        ),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_anthropic_alongside_litellm_when_both_are_available(monkeypatch, tmp_path):
    # Regression test: direct_sdks and model_gateway are orthogonal facts about a
    # pipeline (see docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md), so a
    # pipeline that has both litellm and the anthropic SDK installed must get both
    # adapters wired, not just one.
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(litellm_adapter, "register_litellm_callbacks", lambda sink: captured.update(litellm=sink))
    monkeypatch.setattr(anthropic_adapter, "instrument_anthropic", lambda sink: captured.update(anthropic=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(),
            model_gateway="litellm",
            vector_dbs=frozenset(),
            direct_sdks=frozenset({"anthropic"}),
        ),
    )

    bootstrap_module.bootstrap()

    assert "litellm" in captured
    assert "anthropic" in captured


def test_bootstrap_registers_openai_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(openai_adapter, "instrument_openai", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset(), direct_sdks=frozenset({"openai"})
        ),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_openai_alongside_litellm_when_both_are_available(monkeypatch, tmp_path):
    # Regression test: same orthogonality as anthropic+litellm above -- a pipeline
    # with both litellm and the openai SDK installed gets both adapters wired. The
    # dedup between them (litellm_adapter.inside_litellm_dispatch) is a runtime
    # concern inside openai_adapter itself, not a wiring-time exclusion here.
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(litellm_adapter, "register_litellm_callbacks", lambda sink: captured.update(litellm=sink))
    monkeypatch.setattr(openai_adapter, "instrument_openai", lambda sink: captured.update(openai=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(),
            model_gateway="litellm",
            vector_dbs=frozenset(),
            direct_sdks=frozenset({"openai"}),
        ),
    )

    bootstrap_module.bootstrap()

    assert "litellm" in captured
    assert "openai" in captured


def test_bootstrap_registers_gemini_instrumentation_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(gemini_adapter, "instrument_gemini", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset(), direct_sdks=frozenset({"gemini"})
        ),
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
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"pinecone"})),
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
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"weaviate"})),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_chroma_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_chroma_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"chroma"})),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_qdrant_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_qdrant_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"qdrant"})),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_milvus_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_milvus_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"milvus"})),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_lancedb_callbacks_when_detected(monkeypatch, tmp_path):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(vector_query_adapter, "register_lancedb_callbacks", lambda sink: captured.update(sink=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"lancedb"})),
    )

    bootstrap_module.bootstrap()

    assert "sink" in captured


def test_bootstrap_registers_pinecone_and_chroma_together_when_both_are_available(monkeypatch, tmp_path):
    # Regression test: vector_dbs is a frozenset, not a single exclusive value -- a
    # pipeline with two vector DBs installed (e.g. Pinecone in prod, Chroma for local
    # dev/testing) must get both adapters wired, not just the first match.
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("CASCAID_RUN_ID", "run-1")
    monkeypatch.setenv("CASCAID_EVENTS_PATH", str(events_path))

    captured = {}
    monkeypatch.setattr(
        vector_query_adapter, "register_pinecone_callbacks", lambda sink: captured.update(pinecone=sink)
    )
    monkeypatch.setattr(vector_query_adapter, "register_chroma_callbacks", lambda sink: captured.update(chroma=sink))

    from cascaid.ingestion.stack_detector import DetectedStack

    monkeypatch.setattr(
        "cascaid.ingestion.stack_detector.detect_stack",
        lambda: DetectedStack(
            orchestrators=frozenset(), model_gateway=None, vector_dbs=frozenset({"pinecone", "chroma"})
        ),
    )

    bootstrap_module.bootstrap()

    assert "pinecone" in captured
    assert "chroma" in captured
