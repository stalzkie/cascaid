"""Extracts agent topology from an AutoGen (`autogen-agentchat`) team and tracks
runtime execution (PRD section 4.5). See docs/adr/0006-autogen-agentchat-not-ag2.md for
why this targets `autogen-agentchat` specifically, not the `ag2` fork -- both are live,
different packages under the same "AutoGen" name.

Two seams, chosen after verifying against the real installed library (not assumed from
memory or the package name):

- Static topology + step tracking: `BaseGroupChat.__init__` (every team subclass --
  RoundRobinGroupChat/SelectorGroupChat/Swarm/MagenticOneGroupChat/GraphFlow -- stores its
  constructor's `participants` as `self._participants`/`self._participant_names` here) and
  `BaseGroupChat.run_stream` (an async generator; `run()` itself just does `async for
  message in self.run_stream(...)`, so patching `run_stream` alone covers both entry
  points).
- Per-agent runtime tracking: `ChatAgentContainer.handle_request`
  (`teams/_group_chat/_chat_agent_container.py`) -- the one seam every team type
  dispatches each participant's actual turn through, regardless of which team subclass is
  orchestrating (confirmed by reading every built-in team's source). Patching each team
  type's internal selection logic separately would be exactly the kind of fragile
  per-framework surface this project avoids elsewhere (see langgraph_adapter.py,
  crewai_adapter.py).

`handle_request` is registered as a message handler via autogen_core's `@event` decorator,
which stores routing metadata (`is_message_handler`/`target_types`/`router`) as plain
attributes on the function object and is re-discovered per-instance in
`RoutedAgent.__init__` via a fresh `getattr(cls, name)` -- not baked into a static registry
at class-definition time. Verified empirically before writing this: a bare replacement
function silently becomes an "unhandled message" (the routing table finds no handler and
falls through to `on_unhandled_message`), while `functools.wraps` -- which copies
`__dict__`, where `@event` stores those markers -- is discovered correctly.

Also verified empirically (see the ADR): a contextvar set before `run()`/`run_stream()`
correctly survives into every participant's turn, unlike LangGraph's `ainvoke`, CrewAI's
`async_execution` raw thread, or litellm's background dispatch -- `autogen_core`'s
SingleThreadedAgentRuntime dispatches each turn as a plain awaited coroutine within the
same asyncio task, which Python's contextvars propagate through by default. No CrewAI-style
stash-on-object workaround needed.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable

from cascaid.ingestion.runtime_context import track_node, track_step
from cascaid.ingestion.schema import NodeType

TopologySink = Callable[[dict[str, NodeType], list[tuple[str, str]]], None]


def extract_static_topology(team) -> tuple[dict[str, NodeType], list[tuple[str, str]]]:
    """One AGENT node per participant (a ChatAgent or a nested Team -- both expose
    `.name`; a sub-team is tracked as a single node for its whole turn, not recursed into).
    Edges: a sequential chain in declaration order -- matches RoundRobinGroupChat's actual
    default turn-taking exactly, and approximates SelectorGroupChat/Swarm/
    MagenticOneGroupChat's dynamic selection with the same participant set. Known
    simplification, same class as crewai_adapter.py's own default-process approximation:
    GraphFlow's real DiGraph edges aren't extracted here, it gets this same chain."""
    names = team._participant_names
    nodes: dict[str, NodeType] = {name: NodeType.AGENT for name in names}
    edges = [(names[i - 1], names[i]) for i in range(1, len(names))]
    return nodes, edges


_topology_sink: TopologySink | None = None
_patched = False


def instrument_autogen(topology_sink: TopologySink) -> None:
    """Monkey-patches autogen-agentchat (PRD 4.5) so a customer's pipeline needs zero code
    changes: BaseGroupChat.__init__ extracts topology once per team construction,
    BaseGroupChat.run_stream wraps each run in a fresh track_step (run() calls run_stream()
    internally, so this covers both entry points), and ChatAgentContainer.handle_request
    wraps each participant's actual turn in track_node. The class patches apply once per
    process; topology_sink can be updated on repeat calls without re-patching.

    Known simplification for this pass: nested Team participants are tracked as a single
    node for their whole turn, not recursed into -- matches extract_static_topology's
    simplification above."""
    global _topology_sink, _patched
    _topology_sink = topology_sink
    if _patched:
        return
    _patched = True

    from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat
    from autogen_agentchat.teams._group_chat._chat_agent_container import ChatAgentContainer

    original_init = BaseGroupChat.__init__
    original_run_stream = BaseGroupChat.run_stream
    original_handle_request = ChatAgentContainer.handle_request
    step_counter = itertools.count()

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _topology_sink is not None:
            nodes, edges = extract_static_topology(self)
            _topology_sink(nodes, edges)

    async def patched_run_stream(self, *args, **kwargs):
        with track_step(next(step_counter)):
            async for item in original_run_stream(self, *args, **kwargs):
                yield item

    # functools.wraps is load-bearing here, not cosmetic -- see module docstring: it
    # copies __dict__, where autogen_core's @event decorator stores the routing metadata
    # RoutedAgent.__init__ re-discovers this handler by. Without it, this silently stops
    # being recognized as a message handler at all.
    @functools.wraps(original_handle_request)
    async def patched_handle_request(self, message, ctx):
        with track_node(self._agent.name):
            return await original_handle_request(self, message, ctx)

    BaseGroupChat.__init__ = patched_init
    BaseGroupChat.run_stream = patched_run_stream
    ChatAgentContainer.handle_request = patched_handle_request
