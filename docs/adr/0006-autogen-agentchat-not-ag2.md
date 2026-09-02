---
status: accepted
---

# Target `autogen-agentchat`, not the `ag2` fork, for the AutoGen orchestrator adapter

"AutoGen" is ambiguous today. Microsoft's original `pyautogen` package (last real release
`0.10.0`) was superseded by a ground-up redesign built on `autogen-core`'s actor/
message-passing runtime, shipped as `autogen-agentchat` (actively released, `0.7.5` at
time of writing). A maintainer/community split also produced `ag2` (`1.0.3`), a fork that
continues the pre-redesign `pyautogen` conversational API and is still importable as
`autogen` for backward compatibility. Both are live, both call themselves AutoGen in
spirit, and neither coincides with the PRD's bare "AutoGen" (Section 5.2, 6.1).

Picked `autogen-agentchat`:

- It's Microsoft's own continuation of the AutoGen name and the one new documentation/
  tutorials point to.
- It has an explicit team/graph structure (`teams.GraphFlow` +
  `DiGraphBuilder`/`DiGraph`, alongside the simpler `RoundRobinGroupChat`/
  `SelectorGroupChat`/`Swarm`) — the same "topology is already explicit in code" property
  the PRD calls out as LangGraph/CrewAI's advantage over generic trace-based ingestion
  (Section 5.2). `ag2`'s conversational `GroupChat` model is closer to free-form message
  passing with much less declared structure to extract statically.

Verified against the real installed package (`.venv/Lib/site-packages/autogen_agentchat`)
before writing any adapter code, per this project's standing practice — not assumed from
memory or the package name alone.

## What the real API looks like (verified by introspection + a standalone repro, not guessed)

- Every team type (`RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`,
  `MagenticOneGroupChat`, `GraphFlow`) subclasses `BaseGroupChat`, which stores its
  constructor's `participants: list[ChatAgent | Team]` as `self._participants`/
  `self._participant_names`.
- `BaseGroupChat.run()` is a thin wrapper that does `async for message in
  self.run_stream(...)` — patching `run_stream` alone (an async generator, not a plain
  coroutine) covers both entry points.
- Each participant's actual turn — regardless of which team type is orchestrating —
  dispatches through one shared seam: `ChatAgentContainer.handle_request`
  (`teams/_group_chat/_chat_agent_container.py`), an `autogen_core` actor method decorated
  with `@event`. This is the equivalent of LangGraph's per-node function call or CrewAI's
  `Task._execute_core` — the one place to hook for per-agent runtime tracking without
  patching every team subclass's internal selection logic separately.
- `@event`-decorated methods carry their routing metadata
  (`is_message_handler`/`target_types`/`router`) as plain attributes on the function
  object, re-discovered fresh per instance in `RoutedAgent.__init__` via
  `getattr(cls, name)` — not baked into a static registry at class-definition time.
  Confirmed empirically: replacing `ChatAgentContainer.handle_request` with a bare
  function silently becomes an unhandled message (the routing table finds no handler and
  falls through), while wrapping the original with `functools.wraps` — which copies
  `__dict__`, where `@event` stores those markers — is discovered correctly and dispatches
  as expected.
- Verified empirically (standalone repro: a `RoundRobinGroupChat` of two fake-model
  `AssistantAgent`s, a probe `ContextVar` set before `team.run()`, read from inside a
  patched `handle_request`) that a contextvar set before `run()`/`run_stream()` correctly
  survives into every participant's turn. Unlike LangGraph's `ainvoke`, CrewAI's
  `async_execution` raw thread, or litellm's background dispatch — all of which silently
  drop contextvars across an implicit thread/task boundary — `autogen_core`'s
  `SingleThreadedAgentRuntime` dispatches each turn as a plain awaited coroutine within the
  same asyncio task, which Python propagates context through by default. No CrewAI-style
  stash-on-object workaround needed here.

## Considered Options

- **Target `ag2`** (rejected): still live and still importable as `autogen`, but its
  conversational `GroupChat` model has less explicit, extractable topology than
  `autogen-agentchat`'s team/graph structure, and it's the continuation of the
  pre-redesign API rather than where Microsoft's own AutoGen development is happening.
- **Support both packages** (rejected for this round): real scope increase — two
  different runtimes with different topology shapes and, likely, different context-
  propagation behavior each needing its own empirical verification. Ship one adapter
  correctly first; revisit if a customer actually needs `ag2` specifically.

## Consequences

- `stack_detector.py`'s `ORCHESTRATOR_MODULES` gains `"autogen": "autogen_agentchat"` —
  a pipeline using `ag2`/legacy `pyautogen` (importable as `autogen`) will not be
  auto-detected and falls back to the manual-tracking SDK (ADR 0002), same as any other
  unrecognized orchestrator, not a regression from today's zero AutoGen support.
- `autogen_adapter.py`'s topology extraction only sees participants declared at team
  construction — a nested `Team` participant is tracked as a single node for its whole
  turn, not recursed into, and `GraphFlow`'s real `DiGraph` edges aren't extracted (every
  team type gets the same sequential-chain approximation `crewai_adapter.py` already uses
  for its own default process). Documented simplification, not a correctness bug.
