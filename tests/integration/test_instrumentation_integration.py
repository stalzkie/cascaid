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

import asyncio
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
    litellm.callbacks = []
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
        # Must clear all three registries register_litellm_callbacks populates
        # -- otherwise this test's logger stays registered and keeps firing
        # (accumulating with every other test that does the same) for the
        # rest of the process.
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert topology["nodes"] == {"retriever": NodeType.AGENT, "researcher": NodeType.AGENT}
    assert topology["edges"] == [("retriever", "researcher")]

    assert {e.run_id for e in events} == {run_id}
    assert {e.caller for e in events} == {"researcher"}  # only researcher calls litellm
    assert {e.callee for e in events} == {"gpt-4o-mini"}
    # Two invokes on the same compiled graph -> two distinct steps, proving step
    # tracks "one top-level invocation" and isn't shared/stale across calls.
    assert len({e.step for e in events}) == 2


def _build_async_pipeline():
    async def _retriever(state, config):
        return {"query": state["query"]}

    async def _researcher(state, config):
        await litellm.acompletion(
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


@pytest.mark.integration
def test_real_async_langgraph_pipeline_runs_end_to_end_with_topology_extracted():
    # Regression coverage for the ainvoke instrumentation bug (see
    # docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md):
    # proves the real, composed stack (async nodes + ainvoke + a patched,
    # real litellm.acompletion call) runs without error in a real multi-node
    # graph. Deliberately does NOT wait on litellm's real async
    # success-logging callback actually firing to inspect the resulting
    # CallEvent: empirically, in this environment, litellm defers that
    # dispatch to its own internal background worker with a multi-second,
    # non-deterministic delay (observed failing even at a 45s timeout across
    # repeat runs) -- a real characteristic of litellm's own async logging,
    # not a Cascaid bug, and not worth a flaky CI test. Correctness of both
    # halves of the actual fix -- ainvoke staying entered for the coroutine
    # body, and the metadata snapshot (read instead of by-then-stale
    # contextvars) being built correctly -- is proven deterministically in
    # test_langgraph_adapter.py and test_litellm_adapter.py.
    topology: dict = {}
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    try:
        instrument_langgraph(topology_sink=lambda nodes, edges: topology.update(nodes=nodes, edges=edges))
        register_litellm_callbacks(sink=lambda event: None)

        run_id = str(uuid.uuid4())
        with track_run(run_id):
            compiled = _build_async_pipeline()
            result_1 = asyncio.run(compiled.ainvoke({"query": "what happened to run-42?", "answer": ""}))
            result_2 = asyncio.run(compiled.ainvoke({"query": "second question", "answer": ""}))
    finally:
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []

    assert result_1["answer"] == "done"
    assert result_2["answer"] == "done"
    assert topology["nodes"] == {"retriever": NodeType.AGENT, "researcher": NodeType.AGENT}
    assert topology["edges"] == [("retriever", "researcher")]
