import threading

import pytest
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

import cascaid.ingestion.crewai_adapter as crewai_adapter
from cascaid.ingestion.crewai_adapter import extract_static_topology, instrument_crewai
from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step, track_run
from cascaid.ingestion.schema import NodeType


@pytest.fixture(autouse=True)
def _reset_crewai_patch_state():
    # instrument_crewai() patches Crew.kickoff/Task._execute_core once per process
    # (see its _patched guard) -- without undoing that between tests, whichever
    # test runs first would leave every later test (in this file or any other)
    # calling into a stale patched_kickoff/patched_execute_core closure, or a test
    # that reassigns Crew.kickoff directly would silently strip instrumentation
    # off for everyone after it.
    original_kickoff = Crew.kickoff
    original_execute_core = Task._execute_core
    crewai_adapter._patched = False
    crewai_adapter._topology_sink = None
    yield
    Crew.kickoff = original_kickoff
    Task._execute_core = original_execute_core
    crewai_adapter._patched = False
    crewai_adapter._topology_sink = None


class _FakeLLM(BaseLLM):
    def call(self, *args, **kwargs):
        return "fake response"


@tool("search")
def _search(query: str) -> str:
    """Searches things."""
    return query


def _agent(role: str, tools=None) -> Agent:
    return Agent(
        role=role, goal="test goal", backstory="test backstory", tools=tools or [], llm=_FakeLLM(model="test-fake")
    )


def _task(description: str, agent: Agent, **kwargs) -> Task:
    return Task(description=description, expected_output="an output", agent=agent, **kwargs)


def _build_crew(tasks: list[Task]) -> Crew:
    agents = []
    seen = set()
    for t in tasks:
        if t.agent is not None and id(t.agent) not in seen:
            seen.add(id(t.agent))
            agents.append(t.agent)
    return Crew(agents=agents, tasks=tasks)


def test_extracts_one_node_per_task_and_sequential_chain_edges():
    researcher = _agent("researcher")
    writer = _agent("writer")
    t1 = _task("research", researcher)
    t2 = _task("write", writer)
    crew = _build_crew([t1, t2])

    nodes, edges = extract_static_topology(crew)

    assert nodes == {"researcher (0)": NodeType.AGENT, "writer (1)": NodeType.AGENT}
    assert edges == [("researcher (0)", "writer (1)")]


def test_uses_task_name_when_set():
    researcher = _agent("researcher")
    t1 = _task("research", researcher, name="gather-notes")
    crew = _build_crew([t1])

    nodes, edges = extract_static_topology(crew)

    assert nodes == {"gather-notes": NodeType.AGENT}
    assert edges == []


def test_extracts_tool_nodes_and_edges_from_the_assigned_agent():
    researcher = _agent("researcher", tools=[_search])
    t1 = _task("research", researcher)
    crew = _build_crew([t1])

    nodes, edges = extract_static_topology(crew)

    assert nodes == {"researcher (0)": NodeType.AGENT, "search": NodeType.TOOL}
    assert edges == [("researcher (0)", "search")]


def test_explicit_context_overrides_sequential_chaining():
    a, b, c = _agent("a"), _agent("b"), _agent("c")
    t_a = _task("a", a)
    t_b = _task("b", b)
    t_c = _task("c", c, context=[t_a])
    crew = _build_crew([t_a, t_b, t_c])

    nodes, edges = extract_static_topology(crew)

    assert set(edges) == {("a (0)", "b (1)"), ("a (0)", "c (2)")}


def test_explicit_empty_context_means_no_upstream_edge_not_a_fallback_chain():
    a = _agent("a")
    b = _agent("b")
    t_a = _task("a", a)
    t_b = _task("b", b, context=[])
    crew = _build_crew([t_a, t_b])

    nodes, edges = extract_static_topology(crew)

    assert edges == []


def test_instrument_crewai_wraps_kickoff_in_track_step_and_tasks_in_track_node():
    captured = {}
    agent = _agent("capturer")
    task = _task("d", agent)

    def _fake_orchestration(self, *args, **kwargs):
        captured["step"] = current_step.get()
        captured["node_before_task"] = current_node.get()
        return task._execute_core(agent, "", [])

    def _fake_execute_core(self, agent, context, tools):
        captured["node"] = current_node.get()
        return "ok"

    Crew.kickoff = _fake_orchestration
    Task._execute_core = _fake_execute_core

    instrument_crewai(topology_sink=lambda nodes, edges: None)

    crew = _build_crew([task])
    crew.kickoff()

    assert captured["step"] == 0
    assert captured["node_before_task"] is None
    assert captured["node"] == "capturer (0)"


def test_instrument_crewai_attributes_async_execution_tasks_correctly():
    # Regression test: CrewAI's real Task.execute_async (used whenever a task
    # has async_execution=True, a real documented CrewAI feature -- see
    # crew.py's `if task.async_execution:` branch) spawns a raw
    # threading.Thread, not asyncio.to_thread -- contextvars set during
    # patched_kickoff on the main thread (track_run, track_step) do NOT
    # propagate into that thread. Verified against real CrewAI before writing
    # this fix (see docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md):
    # without it, the async task's run_id/step read as None (dropping every
    # LiteLLM/vector-DB CallEvent it makes) and its node name silently falls
    # back to _task_node_name(task, 0) -- wrong for any async task not at
    # index 0, not just missing.
    captured = {}
    agent0, agent1 = _agent("agent0"), _agent("agent1")
    task0, task1 = _task("task0", agent0), _task("task1", agent1)

    def _fake_orchestration(self, *args, **kwargs):
        # task1 (index 1) executes on a separate raw thread, exactly how
        # CrewAI's real Task.execute_async does for async_execution=True.
        thread = threading.Thread(target=lambda: task1._execute_core(agent1, "", []))
        thread.start()
        thread.join()
        return "ok"

    def _fake_execute_core(self, agent, context, tools):
        captured["run_id"] = current_run_id.get()
        captured["step"] = current_step.get()
        captured["node"] = current_node.get()
        return "ok"

    Crew.kickoff = _fake_orchestration
    Task._execute_core = _fake_execute_core

    instrument_crewai(topology_sink=lambda nodes, edges: None)

    crew = _build_crew([task0, task1])
    with track_run("run-1"):
        crew.kickoff()

    assert captured["run_id"] == "run-1"
    assert captured["step"] == 0
    assert captured["node"] == "agent1 (1)"


def test_instrument_crewai_keeps_task_names_isolated_across_concurrent_kickoffs():
    # Regression test: task_names must not be a single dict cleared/repopulated by
    # every kickoff -- two crews kicked off concurrently on different threads would
    # otherwise stomp on each other's task-name lookups mid-execution. Each crew's
    # task-under-test sits at index 1, not 0: a corrupted lookup falls back to
    # _task_node_name(task, 0), which must be distinguishable from the correct
    # "(1)" name, not coincidentally equal to it.
    agent_a0, agent_a = _agent("agent-a0"), _agent("agent-a")
    agent_b0, agent_b = _agent("agent-b0"), _agent("agent-b")
    task_a0, task_a = _task("task a0", agent_a0), _task("task a", agent_a)
    task_b0, task_b = _task("task b0", agent_b0), _task("task b", agent_b)
    barrier = threading.Barrier(2)
    captured = {}

    crew_a = _build_crew([task_a0, task_a])
    crew_b = _build_crew([task_b0, task_b])

    def _fake_orchestration(self, *args, **kwargs):
        # Forces both kickoffs to have populated their own task_names before either
        # one executes a task, so a shared/cleared dict would be caught red-handed.
        barrier.wait()
        task, agent = (task_a, agent_a) if self is crew_a else (task_b, agent_b)
        return task._execute_core(agent, "", [])

    def _fake_execute_core(self, agent, context, tools):
        captured[agent.role] = current_node.get()
        return "ok"

    Crew.kickoff = _fake_orchestration
    Task._execute_core = _fake_execute_core

    instrument_crewai(topology_sink=lambda nodes, edges: None)

    t1 = threading.Thread(target=crew_a.kickoff)
    t2 = threading.Thread(target=crew_b.kickoff)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert captured["agent-a"] == "agent-a (1)"
    assert captured["agent-b"] == "agent-b (1)"
