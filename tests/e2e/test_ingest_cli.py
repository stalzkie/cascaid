"""E2E seam: cascaid ingest -- closes the persistence gap flagged at the end of
step 3 (Auto-Instrumentation Glue Layer Plan): `cascaid run`'s live JSON-lines
event log reaches the Graph Store + score history, the same way seed_store.py
already does for demo data, so a beta tester's real pipeline shows up in the
dashboard. Sources its events from the *real* bootstrap sinks driving a real
LangGraph+LiteLLM pipeline (same shape as
tests/integration/test_instrumentation_integration.py), not hand-written JSON."""

from __future__ import annotations

import sys
import time
import uuid
from typing import TypedDict

import litellm
import pytest
from langgraph.graph import END, START, StateGraph

import cascaid.ingest as ingest_cli
import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
from cascaid._instrument_bootstrap import _file_sink, _topology_sink
from cascaid.ingestion.langgraph_adapter import instrument_langgraph
from cascaid.ingestion.litellm_adapter import register_litellm_callbacks
from cascaid.ingestion.runtime_context import track_run
from cascaid.storage.db import make_session_factory


class _State(TypedDict):
    query: str


def _build_pipeline():
    def _researcher(state, config):
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": state["query"]}],
            mock_response="an answer",
        )
        return {}

    g = StateGraph(_State)
    g.add_node("researcher", _researcher)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", END)
    return g.compile()


def _wait_for(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("timed out waiting for litellm's async success callback to write events")


@pytest.mark.e2e
def test_ingest_writes_snapshots_and_score_history_from_a_live_events_file(tmp_path, monkeypatch):
    # 1. Train a tiny throwaway model FIRST, before any instrumentation is
    # registered -- instrument_langgraph's patch is process-global and
    # idempotent-once-set, so registering it before run_scenarios (which
    # compiles the demo's own LangGraph pipeline internally) would sweep that
    # unrelated topology into this test's events file too.
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"
    monkeypatch.setattr(
        sys, "argv", ["run_scenarios", "--runs-per-scenario", "2", "--steps", "15", "--out", str(data_dir)]
    )
    run_scenarios_cli.main()
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data_dir), "--epochs", "2", "--out", str(model_path)])
    train_cli.main()

    # 2. Produce a real live events file via the real bootstrap sinks + a real pipeline.
    events_path = tmp_path / "live.jsonl"
    litellm.success_callback = []
    litellm.failure_callback = []
    run_id = str(uuid.uuid4())
    try:
        instrument_langgraph(topology_sink=_topology_sink(str(events_path)))
        register_litellm_callbacks(sink=_file_sink(str(events_path)))
        with track_run(run_id):
            compiled = _build_pipeline()
            compiled.invoke({"query": "q1"})
            compiled.invoke({"query": "q2"})
        _wait_for(lambda: events_path.exists() and len(events_path.read_text(encoding="utf-8").splitlines()) >= 3)
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []

    # 3. cascaid ingest: events file -> Graph Store snapshots + score history.
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--events",
            str(events_path),
            "--store",
            str(store_dir),
            "--model",
            str(model_path),
            "--database-url",
            database_url,
        ],
    )
    ingest_cli.main()

    from cascaid.ingestion.graph_store import list_snapshots

    snapshots = list_snapshots(store_dir, run_id)
    assert len(snapshots) == 2  # two compiled.invoke() calls -> two steps

    from cascaid.storage.repository import get_score_history

    with make_session_factory(database_url)() as session:
        history = get_score_history(session, run_id=run_id)
    assert len(history) > 0
    assert {row.node_name for row in history} == {"researcher"}
