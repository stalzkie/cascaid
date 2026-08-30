"""Extracts agent/task topology from a CrewAI Crew and tracks runtime execution
(PRD section 4.5). Patches Crew.kickoff and Task._execute_core directly rather than
CrewAI's event bus: the event bus dispatches sync handlers via a ThreadPoolExecutor
over a *copied* contextvars.Context (see crewai.events.event_bus.emit), so a
ContextVar set from an event handler never becomes visible in the thread actually
running the task/LLM call it's meant to tag -- verified against CrewAI's source
before choosing this seam over the (more idiomatic-looking) event-listener API.

The same class of problem shows up again one level down: a Task with
`async_execution=True` (crew.py's `if task.async_execution:` branch) runs via
`Task.execute_async`, which spawns a raw `threading.Thread` -- not
`asyncio.to_thread`, which *does* copy contextvars. A raw thread does not
inherit the calling context, so track_step/track_run set during
patched_kickoff (on the main thread) are invisible inside patched_execute_core
when CrewAI runs it on that new thread (verified empirically -- see
docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md): the async
task's step/run_id read as None (dropping every LiteLLM/vector-DB CallEvent it
makes) and its node name silently falls back to _task_node_name(task, 0) --
wrong for any async task not at index 0, not just missing. Fixed by stashing
each task's run_id/step/name directly as attributes on the live Task object at
kickoff time (ordinary attribute access needs no thread-context propagation,
unlike a ContextVar) and having patched_execute_core re-enter
track_run/track_step/track_node itself from whatever thread it actually runs
on, instead of assuming the kickoff-level context is still in scope.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from cascaid.ingestion.runtime_context import current_run_id, track_node, track_run, track_step
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


def instrument_crewai(topology_sink: TopologySink) -> None:
    """Monkey-patches CrewAI (PRD 4.5) so a customer's pipeline needs zero code
    changes: Crew.kickoff extracts topology once per call and stashes each
    task's step/node-name directly onto the live Task object (see module
    docstring for why: a ContextVar wouldn't survive Task.execute_async's raw
    thread), and Task._execute_core -- the shared core execute_sync/
    execute_async both call, on whichever thread CrewAI actually runs it on --
    reads that stashed step/name back off `self` and re-enters
    track_step/track_node itself. The class patch applies once per process;
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
        step = next(step_counter)
        # Captured here (main thread, where it's already correctly set by
        # whatever wrapped this kickoff -- e.g. the bootstrap's track_run),
        # not read fresh in patched_execute_core: current_run_id is exactly as
        # vulnerable to Task.execute_async's raw-thread context loss as
        # current_step is (verified empirically, same as the docstring above).
        run_id = current_run_id.get()
        for task, name in zip(self.tasks, _task_node_names(self.tasks), strict=True):
            task._cascaid_step = step
            task._cascaid_name = name
            task._cascaid_run_id = run_id
        with track_step(step):
            return original_kickoff(self, *args, **kwargs)

    def patched_execute_core(self, agent, context, tools):
        name = getattr(self, "_cascaid_name", None) or _task_node_name(self, 0)
        step = getattr(self, "_cascaid_step", None)
        run_id = getattr(self, "_cascaid_run_id", None)
        with track_run(run_id), track_step(step), track_node(name):
            return original_execute_core(self, agent, context, tools)

    Crew.kickoff = patched_kickoff
    Task._execute_core = patched_execute_core
