"""Extracts agent/task topology from a CrewAI Crew and tracks runtime execution
(PRD section 4.5). Patches Crew.kickoff and Task._execute_core directly rather than
CrewAI's event bus: the event bus dispatches sync handlers via a ThreadPoolExecutor
over a *copied* contextvars.Context (see crewai.events.event_bus.emit), so a
ContextVar set from an event handler never becomes visible in the thread actually
running the task/LLM call it's meant to tag -- verified against CrewAI's source
before choosing this seam over the (more idiomatic-looking) event-listener API.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from cascaid.ingestion.runtime_context import track_node, track_step
from cascaid.ingestion.schema import NodeType

TopologySink = Callable[[dict[str, NodeType], list[tuple[str, str]]], None]


def _task_node_name(task, index: int) -> str:
    if task.name:
        return task.name
    role = task.agent.role if task.agent is not None else "unassigned"
    return f"{role} ({index})"


def extract_static_topology(crew) -> tuple[dict[str, NodeType], list[tuple[str, str]]]:
    """One node per Task (typed AGENT -- the schema has no distinct task type, and a
    task's execution is the unit Cascaid needs to track risk against), plus one TOOL
    node per distinct tool a task's agent can call. Edges: each task's declared
    `context` (upstream tasks whose output feeds it); a task with no explicit context
    chains from the previous task in Crew.tasks order, matching CrewAI's own default
    sequential-process behavior. Known simplification: Process.hierarchical's
    manager-agent delegation isn't modeled as its own edge."""
    task_names = [_task_node_name(task, i) for i, task in enumerate(crew.tasks)]
    nodes: dict[str, NodeType] = {}
    edges: list[tuple[str, str]] = []

    for i, task in enumerate(crew.tasks):
        name = task_names[i]
        nodes[name] = NodeType.AGENT

        tools = task.tools or (task.agent.tools if task.agent is not None else None) or []
        for t in tools:
            nodes[t.name] = NodeType.TOOL
            edges.append((name, t.name))

        context = task.context if isinstance(task.context, list) else None
        if context:
            for upstream in context:
                edges.append((task_names[crew.tasks.index(upstream)], name))
        elif i > 0:
            edges.append((task_names[i - 1], name))

    return nodes, edges


_topology_sink: TopologySink | None = None
_patched = False


def instrument_crewai(topology_sink: TopologySink) -> None:
    """Monkey-patches CrewAI (PRD 4.5) so a customer's pipeline needs zero code
    changes: Crew.kickoff extracts topology once per call and wraps the call in a
    fresh track_step (one step = one top-level kickoff, matching what the GNN was
    trained on), and Task._execute_core -- the shared core execute_sync/execute_async
    both call -- wraps each task's execution in track_node using the same node name
    extract_static_topology assigned it. The class patch applies once per process;
    topology_sink can be updated on repeat calls without re-patching.

    Known simplification: only Crew.kickoff is wrapped, not kickoff_for_each; only
    the common Task._execute_core call shape is patched."""
    global _topology_sink, _patched
    _topology_sink = topology_sink
    if _patched:
        return
    _patched = True

    from crewai import Crew, Task

    original_kickoff = Crew.kickoff
    original_execute_core = Task._execute_core
    step_counter = itertools.count()
    task_names: dict[int, str] = {}

    def patched_kickoff(self, *args, **kwargs):
        if _topology_sink is not None:
            nodes, edges = extract_static_topology(self)
            _topology_sink(nodes, edges)
        task_names.clear()
        task_names.update({id(task): _task_node_name(task, i) for i, task in enumerate(self.tasks)})
        with track_step(next(step_counter)):
            return original_kickoff(self, *args, **kwargs)

    def patched_execute_core(self, agent, context, tools):
        name = task_names.get(id(self)) or _task_node_name(self, 0)
        with track_node(name):
            return original_execute_core(self, agent, context, tools)

    Crew.kickoff = patched_kickoff
    Task._execute_core = patched_execute_core
