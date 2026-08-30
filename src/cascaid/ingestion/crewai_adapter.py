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
from contextvars import ContextVar

from cascaid.ingestion.runtime_context import track_node, track_step
from cascaid.ingestion.schema import NodeType

TopologySink = Callable[[dict[str, NodeType], list[tuple[str, str]]], None]


def _task_node_name(task, index: int) -> str:
    if task.name:
        return task.name
    role = task.agent.role if task.agent is not None else "unassigned"
    return f"{role} ({index})"


def _task_node_names(tasks) -> list[str]:
    return [_task_node_name(task, i) for i, task in enumerate(tasks)]


def extract_static_topology(crew) -> tuple[dict[str, NodeType], list[tuple[str, str]]]:
    """One node per Task (typed AGENT -- the schema has no distinct task type, and a
    task's execution is the unit Cascaid needs to track risk against), plus one TOOL
    node per distinct tool a task's agent can call. Edges: each task's declared
    `context` (upstream tasks whose output feeds it) when it's an explicit list --
    an explicit empty list means no upstream edge, not a fallback. A task whose
    context is unset entirely chains from the previous task in Crew.tasks order,
    matching CrewAI's own default sequential-process behavior. Known
    simplification: Process.hierarchical's manager-agent delegation isn't modeled
    as its own edge."""
    task_names = _task_node_names(crew.tasks)
    nodes: dict[str, NodeType] = {}
    edges: list[tuple[str, str]] = []

    for i, task in enumerate(crew.tasks):
        name = task_names[i]
        nodes[name] = NodeType.AGENT

        tools = task.tools or (task.agent.tools if task.agent is not None else None) or []
        for t in tools:
            nodes[t.name] = NodeType.TOOL
            edges.append((name, t.name))

        # `task.context` is `list[Task] | None | _NotSpecified` -- an explicit `[]`
        # ("no upstream context") is a list too, so it must NOT fall through to the
        # sequential-chain default the way an unset/None context does.
        has_explicit_context = isinstance(task.context, list)
        if has_explicit_context:
            for upstream in task.context:
                edges.append((task_names[crew.tasks.index(upstream)], name))
        elif i > 0:
            edges.append((task_names[i - 1], name))

    return nodes, edges


_topology_sink: TopologySink | None = None
_patched = False

# Per-kickoff, not a shared module-level dict: two Crew.kickoff() calls running
# concurrently on different threads each need their own view of "which task name
# goes with which Task object" without clobbering each other's, the same isolation
# problem track_node/track_step already solve for -- so this reuses that mechanism
# rather than inventing a second one.
_task_names: ContextVar[dict[int, str] | None] = ContextVar("_crewai_task_names", default=None)


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

    def patched_kickoff(self, *args, **kwargs):
        if _topology_sink is not None:
            nodes, edges = extract_static_topology(self)
            _topology_sink(nodes, edges)
        # Keyed by id(task): Task has no natural hashable business key, and this is
        # safe because self.tasks holds live references for the whole call below --
        # nothing here can be garbage-collected (and have its id() reused) while
        # in scope.
        names = {id(task): name for task, name in zip(self.tasks, _task_node_names(self.tasks), strict=True)}
        token = _task_names.set(names)
        try:
            with track_step(next(step_counter)):
                return original_kickoff(self, *args, **kwargs)
        finally:
            _task_names.reset(token)

    def patched_execute_core(self, agent, context, tools):
        names = _task_names.get()
        name = (names.get(id(self)) if names else None) or _task_node_name(self, 0)
        with track_node(name):
            return original_execute_core(self, agent, context, tools)

    Crew.kickoff = patched_kickoff
    Task._execute_core = patched_execute_core
