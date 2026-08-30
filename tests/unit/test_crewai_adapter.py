from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool

from cascaid.ingestion.crewai_adapter import extract_static_topology, instrument_crewai
from cascaid.ingestion.runtime_context import current_node, current_step
from cascaid.ingestion.schema import NodeType


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
