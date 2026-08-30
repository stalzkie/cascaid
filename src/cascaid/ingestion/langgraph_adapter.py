"""Extracts agent/tool topology from a compiled LangGraph app (PRD section 4.5, static seam).

Model-endpoint/vector-store nodes and their edges are not LangGraph nodes at all -- they
only appear when a node's code calls out to LiteLLM/a vector DB client at runtime, so
they're populated by a separate runtime-observation seam, not this one.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from cascaid.ingestion.runtime_context import track_node, track_step
from cascaid.ingestion.schema import NodeType

_SENTINEL_NODES = {"__start__", "__end__"}

TopologySink = Callable[[dict[str, NodeType], list[tuple[str, str]]], None]


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


def _wrap_node_fn(name: str, fn):
    # Runnables/tools are always called via a uniform .invoke(input, config) --
    # no signature-introspection risk. Plain functions are what LangGraph inspects
    # (via inspect.signature) to decide whether to pass `config`, so that wrapper
    # needs functools.wraps to keep exposing fn's real signature through __wrapped__.
    if hasattr(fn, "invoke"):

        def wrapped(*args, **kwargs):
            with track_node(name):
                return fn.invoke(*args, **kwargs)

        return wrapped

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with track_node(name):
            return fn(*args, **kwargs)

    return wrapped


def _wrap_invoke(original_invoke, step_counter: itertools.count):
    def wrapped(*args, **kwargs):
        with track_step(next(step_counter)):
            return original_invoke(*args, **kwargs)

    return wrapped


_topology_sink: TopologySink | None = None
_patched = False


def instrument_langgraph(topology_sink: TopologySink) -> None:
    """Monkey-patches LangGraph (PRD 4.5, static + runtime seams) so a customer's
    pipeline needs zero code changes: StateGraph.add_node wraps each node's
    execution in track_node (a targeted seam -- patching LangGraph's internal
    Pregel dispatch loop directly would be far more fragile across versions), and
    StateGraph.compile extracts topology once and wraps invoke()/ainvoke() in a
    fresh track_step per call so "step" keeps meaning one top-level invocation,
    matching what the GNN was trained on (see the Auto-Instrumentation Glue Layer
    Plan). The class patch applies once per process; topology_sink can be updated
    on repeat calls without re-patching.

    Known simplification for this pass: only the common `add_node(name, fn)` call
    shape is wrapped (used by every real usage in this codebase and LangGraph's
    own docs); other call shapes (e.g. name inferred from fn) pass through
    unwrapped rather than risk mis-handling an unfamiliar signature. Likewise only
    a single global step counter is kept, shared across every compiled graph in
    the process -- fine for the one-pipeline-per-`cascaid run` case this targets.
    """
    global _topology_sink, _patched
    _topology_sink = topology_sink
    if _patched:
        return
    _patched = True

    original_add_node = StateGraph.add_node
    original_compile = StateGraph.compile

    def patched_add_node(self, *args, **kwargs):
        if len(args) >= 2 and callable(args[1]):
            name, fn, rest = args[0], args[1], args[2:]
            return original_add_node(self, name, _wrap_node_fn(name, fn), *rest, **kwargs)
        return original_add_node(self, *args, **kwargs)

    def patched_compile(self, *args, **kwargs):
        compiled = original_compile(self, *args, **kwargs)
        if _topology_sink is not None:
            nodes, edges = extract_static_topology(compiled)
            _topology_sink(nodes, edges)

        step_counter = itertools.count()
        compiled.invoke = _wrap_invoke(compiled.invoke, step_counter)
        compiled.ainvoke = _wrap_invoke(compiled.ainvoke, step_counter)
        return compiled

    StateGraph.add_node = patched_add_node
    StateGraph.compile = patched_compile
