"""Manual instrumentation for frameworks Cascaid can't auto-detect (PRD 4.5, runtime
seam): the `ag2` fork of AutoGen (only `autogen-agentchat` is auto-detected -- see
docs/adr/0006-autogen-agentchat-not-ag2.md), hand-rolled orchestration, an in-house
model client. See docs/adr/0002-manual-tracking-sdk-is-context-manager-shaped.md for
why this is a context manager, not a decorator or an explicit-argument function call.

current_run_id is set process-wide by _instrument_bootstrap.py regardless of what's
auto-detected, but current_step is set *only* by langgraph_adapter.py/crewai_adapter.py/
autogen_adapter.py -- for a pipeline with none of those (this module's actual target
case), nothing else ever sets it. observe_call alone would silently no-op every time
it's used for exactly the case it exists for, unless a step boundary is also marked
manually -- see cascaid/__init__.py, which re-exports track_step/track_run alongside
observe_call so a customer marks those boundaries themselves, the same job the
auto-detected orchestrator integrations do automatically for their frameworks.

Unlike every other adapter in this package, nothing in _instrument_bootstrap.py wires
this one up -- it's invoked directly by the customer's own code, not auto-patched onto
a detected library, so there's no single registration point to hand it a sink the way
register_litellm_callbacks/instrument_anthropic get one. It resolves its own sink per
call instead, writing straight to CASCAID_EVENTS_PATH (same file, same JSON-line format
every other adapter's sink writes to via _instrument_bootstrap.py's _file_sink) unless a
sink is passed explicitly -- kept as a small local duplicate of that logic rather than
importing a private helper from _instrument_bootstrap.py, matching every other adapter
module's independence from that module.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step
from cascaid.ingestion.schema import CallEvent, NodeType


class CallTracker:
    def __init__(self):
        self.event: CallEvent | None = None


def _default_sink(event: CallEvent) -> None:
    events_path = os.environ.get("CASCAID_EVENTS_PATH")
    if not events_path:
        return
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "call_event", **event.to_json()}) + "\n")


def _build_call_event(
    caller: str,
    callee: str,
    caller_type: NodeType,
    callee_type: NodeType,
    run_id: str,
    step: int,
    scenario: str,
    error: bool,
    start: float,
) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario=scenario,
        step=step,
        caller=caller,
        callee=callee,
        caller_type=caller_type,
        callee_type=callee_type,
        latency_ms=(time.perf_counter() - start) * 1000,
        error=error,
        retried=False,
        token_cost=0.0,
        occurred_at=datetime.now(timezone.utc),
    )


@contextmanager
def observe_call(
    callee: str,
    callee_type: NodeType,
    *,
    caller: str | None = None,
    caller_type: NodeType = NodeType.AGENT,
    scenario: str = "production",
    sink: Callable[[CallEvent], None] | None = None,
):
    """Records one manually-instrumented call as a CallEvent and sinks it. Skips
    sinking (yields a tracker with `.event` left None, never raises into the
    customer's call path) when current_run_id/current_step aren't set -- e.g. no
    `track_run`/`track_step` block is active yet -- same guard every auto-detected
    adapter uses (see register_litellm_callbacks). `caller` defaults to
    current_node when unset."""
    run_id, step = current_run_id.get(), current_step.get()
    tracker = CallTracker()
    if run_id is None or step is None:
        yield tracker
        return

    resolved_caller = caller if caller is not None else (current_node.get() or "unknown")
    resolved_sink = sink if sink is not None else _default_sink
    start = time.perf_counter()
    error = False
    try:
        yield tracker
    except Exception:
        error = True
        raise
    finally:
        tracker.event = _build_call_event(
            resolved_caller, callee, caller_type, callee_type, run_id, step, scenario, error, start
        )
        resolved_sink(tracker.event)


@asynccontextmanager
async def observe_call_async(
    callee: str,
    callee_type: NodeType,
    *,
    caller: str | None = None,
    caller_type: NodeType = NodeType.AGENT,
    scenario: str = "production",
    sink: Callable[[CallEvent], None] | None = None,
):
    """Async twin of observe_call, for a customer's async call sites."""
    run_id, step = current_run_id.get(), current_step.get()
    tracker = CallTracker()
    if run_id is None or step is None:
        yield tracker
        return

    resolved_caller = caller if caller is not None else (current_node.get() or "unknown")
    resolved_sink = sink if sink is not None else _default_sink
    start = time.perf_counter()
    error = False
    try:
        yield tracker
    except Exception:
        error = True
        raise
    finally:
        tracker.event = _build_call_event(
            resolved_caller, callee, caller_type, callee_type, run_id, step, scenario, error, start
        )
        resolved_sink(tracker.event)
