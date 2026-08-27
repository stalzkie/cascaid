"""Extracts agent/tool topology from a compiled LangGraph app (PRD section 4.5, static seam).

Model-endpoint/vector-store nodes and their edges are not LangGraph nodes at all -- they
only appear when a node's code calls out to LiteLLM/a vector DB client at runtime, so
they're populated by a separate runtime-observation seam, not this one.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from cascaid.ingestion.schema import NodeType

_SENTINEL_NODES = {"__start__", "__end__"}


def extract_static_topology(compiled_graph) -> tuple[dict[str, NodeType], list[tuple[str, str]]]:
    graph = compiled_graph.get_graph()

    nodes = {
        name: (NodeType.TOOL if isinstance(node.data, BaseTool) else NodeType.AGENT)
        for name, node in graph.nodes.items()
        if name not in _SENTINEL_NODES
    }
    edges = [
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.source not in _SENTINEL_NODES and edge.target not in _SENTINEL_NODES
    ]
    return nodes, edges
