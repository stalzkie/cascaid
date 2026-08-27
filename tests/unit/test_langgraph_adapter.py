from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from cascaid.ingestion.langgraph_adapter import extract_static_topology
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
