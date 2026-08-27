"""Builds the static dependency graph from a topology descriptor + reads raw run event logs.

The topology descriptor (topology.json) is what a real ingestion agent would infer from
LangGraph's compiled execution graph (agent/tool control-flow edges) plus observed
LiteLLM/vector-DB service calls (Section 5.2) -- this module is stack-agnostic, it just
consumes whatever topology + events it is handed.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from cascaid.ingestion.schema import CallEvent, NodeType


def load_topology(path: str | Path) -> tuple[dict[str, NodeType], list[tuple[str, str]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {name: NodeType(t) for name, t in raw["nodes"].items()}
    edges = [(e[0], e[1]) for e in raw["edges"]]
    return nodes, edges


def build_static_graph(nodes: dict[str, NodeType], edges: list[tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for name, node_type in nodes.items():
        g.add_node(name, node_type=node_type)
    for caller, callee in edges:
        g.add_edge(caller, callee)
    return g


def load_run_events(path: str | Path) -> list[CallEvent]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(CallEvent.from_json(json.loads(line)))
    return events


def load_manifest(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
