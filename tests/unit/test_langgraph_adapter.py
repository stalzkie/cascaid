import asyncio
from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from cascaid.ingestion.langgraph_adapter import extract_static_topology, instrument_langgraph
from cascaid.ingestion.runtime_context import current_node, current_step
from cascaid.ingestion.schema import NodeType


class _State(TypedDict):
    x: str


@tool
def _search(query: str) -> str:
    """Searches things."""
    return query


def _build_compiled_graph():
    g = StateGraph(_State)
    g.add_node("planner", lambda s: {})
    g.add_node("search", _search)
    g.add_edge(START, "planner")
    g.add_edge("planner", "search")
    g.add_edge("search", END)
    return g.compile()


def test_extracts_node_types_and_control_edges():
    nodes, edges = extract_static_topology(_build_compiled_graph())
    assert nodes == {"planner": NodeType.AGENT, "search": NodeType.TOOL}
    assert edges == [("planner", "search")]


def test_instrument_langgraph_wraps_each_node_execution_in_track_node():
    captured = {}

    def _planner(state, config):
        captured["node_during_planner"] = current_node.get()
        return {}

    def _synth(state, config):
        captured["node_during_synth"] = current_node.get()
        return {}

    instrument_langgraph(topology_sink=lambda nodes, edges: None)

    g = StateGraph(_State)
    g.add_node("planner", _planner)
    g.add_node("synth", _synth)
    g.add_edge(START, "planner")
    g.add_edge("planner", "synth")
    g.add_edge("synth", END)
    compiled = g.compile()

    compiled.invoke({"x": ""})

    assert captured["node_during_planner"] == "planner"
    assert captured["node_during_synth"] == "synth"
    # Proves attribution didn't just latch onto whichever node ran last.
    assert captured["node_during_planner"] != captured["node_during_synth"]


def test_instrument_langgraph_calls_topology_sink_on_compile():
    captured = {}
    instrument_langgraph(topology_sink=lambda nodes, edges: captured.update(nodes=nodes, edges=edges))

    g = StateGraph(_State)
    g.add_node("planner", lambda s: {})
    g.add_node("search", _search)
    g.add_edge(START, "planner")
    g.add_edge("planner", "search")
    g.add_edge("search", END)
    g.compile()

    assert captured["nodes"] == {"planner": NodeType.AGENT, "search": NodeType.TOOL}
    assert captured["edges"] == [("planner", "search")]


def test_instrument_langgraph_wraps_invoke_in_a_fresh_step_per_call():
    captured = []

    def _planner(state, config):
        captured.append(current_step.get())
        return {}

    instrument_langgraph(topology_sink=lambda nodes, edges: None)

    g = StateGraph(_State)
    g.add_node("planner", _planner)
    g.add_edge(START, "planner")
    g.add_edge("planner", END)
    compiled = g.compile()

    compiled.invoke({"x": ""})
    compiled.invoke({"x": ""})

    assert len(captured) == 2
    assert all(step is not None for step in captured)
    assert captured[0] != captured[1]
    assert current_step.get() is None  # resets after each invoke


def test_instrument_langgraph_wraps_ainvoke_and_stays_entered_for_the_coroutine_body():
    # Regression test: a naive sync wrapper around ainvoke would exit its
    # `with track_step(...)` block the instant the coroutine is *constructed*,
    # not once it actually runs -- resetting current_step to None before any
    # node executes, silently breaking every LiteLLM/vector-DB CallEvent for
    # any pipeline invoked via ainvoke (see
    # docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md).
    captured = []

    async def _planner(state, config):
        captured.append(current_step.get())
        return {}

    instrument_langgraph(topology_sink=lambda nodes, edges: None)

    g = StateGraph(_State)
    g.add_node("planner", _planner)
    g.add_edge(START, "planner")
    g.add_edge("planner", END)
    compiled = g.compile()

    asyncio.run(compiled.ainvoke({"x": ""}))
    asyncio.run(compiled.ainvoke({"x": ""}))

    assert len(captured) == 2
    assert all(step is not None for step in captured)
    assert captured[0] != captured[1]
    assert current_step.get() is None  # resets after each ainvoke
