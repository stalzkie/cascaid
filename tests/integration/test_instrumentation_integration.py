"""Integration seam: instrument_langgraph + register_litellm_callbacks wired
together against a real (not hand-built) LangGraph pipeline making real LiteLLM
calls -- catches integration bugs the adapters' isolated unit tests can't (Auto-
Instrumentation Glue Layer Plan, step 3).

Deliberately NOT cascaid_demo/run_scenarios.py: that pipeline's fault injection is
precisely rng-controlled statistics validated against the 0.90 PR-AUC accuracy
number (see docs/GNN_Accuracy_Improvement_Log.md) -- routing it through real
LiteLLM dispatch would risk that number for no integration-correctness benefit.
This is a separate, dedicated proof pipeline instead."""

from __future__ import annotations

import time
import uuid
from typing import TypedDict

import litellm
import pytest
from langgraph.graph import END, START, StateGraph

from cascaid.ingestion.langgraph_adapter import instrument_langgraph
from cascaid.ingestion.litellm_adapter import register_litellm_callbacks
from cascaid.ingestion.runtime_context import track_run
from cascaid.ingestion.schema import NodeType


class _State(TypedDict):
    query: str
    answer: str


def _build_pipeline():
    def _retriever(state, config):
        return {"query": state["query"]}

    def _researcher(state, config):
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": state["query"]}],
            mock_response="a well-researched answer",
        )
        return {"answer": "done"}

    g = StateGraph(_State)
    g.add_node("retriever", _retriever)
    g.add_node("researcher", _researcher)
    g.add_edge(START, "retriever")
    g.add_edge("retriever", "researcher")
    g.add_edge("researcher", END)
    return g.compile()


def _wait_for(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("timed out waiting for litellm's async success callback to fire")


@pytest.mark.integration
def test_real_langgraph_pipeline_produces_correctly_attributed_call_events():
    topology: dict = {}
    events: list = []
    litellm.success_callback = []
    litellm.failure_callback = []
    try:
        instrument_langgraph(topology_sink=lambda nodes, edges: topology.update(nodes=nodes, edges=edges))
        register_litellm_callbacks(sink=events.append)

        run_id = str(uuid.uuid4())
        with track_run(run_id):
            compiled = _build_pipeline()
            compiled.invoke({"query": "what happened to run-42?", "answer": ""})
            compiled.invoke({"query": "second question", "answer": ""})

        _wait_for(lambda: len(events) == 2)
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []

    assert topology["nodes"] == {"retriever": NodeType.AGENT, "researcher": NodeType.AGENT}
    assert topology["edges"] == [("retriever", "researcher")]

    assert {e.run_id for e in events} == {run_id}
    assert {e.caller for e in events} == {"researcher"}  # only researcher calls litellm
    assert {e.callee for e in events} == {"gpt-4o-mini"}
    # Two invokes on the same compiled graph -> two distinct steps, proving step
    # tracks "one top-level invocation" and isn't shared/stale across calls.
    assert len({e.step for e in events}) == 2
