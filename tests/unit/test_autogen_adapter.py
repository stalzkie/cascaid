import asyncio

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat
from autogen_agentchat.teams._group_chat._chat_agent_container import ChatAgentContainer
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

import cascaid.ingestion.autogen_adapter as autogen_adapter
from cascaid.ingestion.autogen_adapter import extract_static_topology, instrument_autogen
from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step, track_run
from cascaid.ingestion.schema import NodeType


@pytest.fixture(autouse=True)
def _reset_autogen_patch_state():
    # instrument_autogen() patches BaseGroupChat.__init__/run_stream and
    # ChatAgentContainer.handle_request once per process (see its _patched guard) --
    # without undoing that between tests, whichever test runs first would leave every
    # later test calling into a stale patched closure.
    original_init = BaseGroupChat.__init__
    original_run_stream = BaseGroupChat.run_stream
    original_handle_request = ChatAgentContainer.handle_request
    autogen_adapter._patched = False
    autogen_adapter._topology_sink = None
    yield
    BaseGroupChat.__init__ = original_init
    BaseGroupChat.run_stream = original_run_stream
    ChatAgentContainer.handle_request = original_handle_request
    autogen_adapter._patched = False
    autogen_adapter._topology_sink = None


class _FakeModelClient(ChatCompletionClient):
    """Minimal fake so tests don't need a real LLM API key -- returns one fixed reply
    per call, enough to drive a RoundRobinGroupChat to a real MaxMessageTermination."""

    def __init__(self):
        self._n = 0

    async def create(self, messages, **kwargs):
        self._n += 1
        return CreateResult(
            finish_reason="stop",
            content=f"reply-{self._n}",
            usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
            cached=False,
        )

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError

    async def close(self):
        pass

    def actual_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, *args, **kwargs):
        return 0

    def remaining_tokens(self, *args, **kwargs):
        return 0

    @property
    def capabilities(self):
        return {"vision": False, "function_calling": False, "json_output": False}

    @property
    def model_info(self):
        return {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        }


def _agent(name: str) -> AssistantAgent:
    return AssistantAgent(name, model_client=_FakeModelClient())


def _build_team(agents, max_messages: int = 4) -> RoundRobinGroupChat:
    return RoundRobinGroupChat(agents, termination_condition=MaxMessageTermination(max_messages))


def test_extracts_one_node_per_participant_and_sequential_chain_edges():
    team = _build_team([_agent("researcher"), _agent("writer")])

    nodes, edges = extract_static_topology(team)

    assert nodes == {"researcher": NodeType.AGENT, "writer": NodeType.AGENT}
    assert edges == [("researcher", "writer")]


def test_single_participant_has_no_edges():
    team = _build_team([_agent("solo")])

    nodes, edges = extract_static_topology(team)

    assert nodes == {"solo": NodeType.AGENT}
    assert edges == []


def test_instrument_autogen_calls_topology_sink_on_team_construction():
    captured = {}

    instrument_autogen(topology_sink=lambda nodes, edges: captured.update(nodes=nodes, edges=edges))

    _build_team([_agent("a"), _agent("b")])

    assert captured["nodes"] == {"a": NodeType.AGENT, "b": NodeType.AGENT}
    assert captured["edges"] == [("a", "b")]


def test_instrument_autogen_tracks_step_and_node_across_real_dispatch():
    # Real end-to-end run through autogen_core's actor runtime (not a faked
    # kickoff/execute_core reassignment like crewai_adapter's tests) -- this class of
    # framework has previously hidden real contextvar-loss bugs behind an implicit
    # thread/task boundary (LangGraph's ainvoke, CrewAI's async_execution thread,
    # litellm's background dispatch -- see
    # docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md), so this needs
    # to prove attribution survives the actual dispatch, not just the wrapper's own
    # frame.
    observed_calls = []

    class _ObservingClient(_FakeModelClient):
        async def create(self, messages, **kwargs):
            observed_calls.append(
                {
                    "run_id": current_run_id.get(),
                    "step": current_step.get(),
                    "node": current_node.get(),
                }
            )
            return await super().create(messages, **kwargs)

    instrument_autogen(topology_sink=lambda nodes, edges: None)

    agent_a = AssistantAgent("agent_a", model_client=_ObservingClient())
    agent_b = AssistantAgent("agent_b", model_client=_ObservingClient())
    team = _build_team([agent_a, agent_b])

    async def _run():
        with track_run("run-1"):
            await team.run(task="hello")

    asyncio.run(_run())

    assert len(observed_calls) >= 2
    assert all(call["run_id"] == "run-1" for call in observed_calls)
    assert all(call["step"] == 0 for call in observed_calls)
    assert observed_calls[0]["node"] == "agent_a"
    assert observed_calls[1]["node"] == "agent_b"


def test_instrument_autogen_increments_step_per_run_call():
    instrument_autogen(topology_sink=lambda nodes, edges: None)

    observed_steps = []

    class _StepObservingClient(_FakeModelClient):
        async def create(self, messages, **kwargs):
            observed_steps.append(current_step.get())
            return await super().create(messages, **kwargs)

    agent = AssistantAgent("solo", model_client=_StepObservingClient())
    team = _build_team([agent], max_messages=2)

    async def _run():
        await team.run(task="first")
        await team.run(task="second")

    asyncio.run(_run())

    assert observed_steps[0] == 0
    assert observed_steps[-1] == 1
