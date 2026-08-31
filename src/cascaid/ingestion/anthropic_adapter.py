"""Direct Anthropic SDK instrumentation (PRD section 4.5, runtime seam): for pipelines
that call anthropic.Anthropic()/AsyncAnthropic() directly rather than through litellm.
See docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md for why this ships
before an equivalent OpenAI adapter.

Unlike litellm_adapter.py, the anthropic SDK has no callback registry to hook into --
the only interception point is patching Messages.create/AsyncMessages.create directly.
That turns out to be simpler in one respect: the wrapped call converts its own response
to a CallEvent and sinks it *inline*, in the same call stack (sync) or coroutine (async)
as the original call -- there is no detached/background dispatch the way litellm defers
its async and streaming callbacks (see litellm_adapter.py's module docstring). So
current_run_id/current_step/current_node can be read live via runtime_context's
contextvars at the point the wrapped create() returns; no pre-dispatch metadata-snapshot
workaround is needed here.

Streaming (client.messages.stream()) is out of scope for this pass -- not wired up, so a
streaming call passes through uninstrumented, the same kind of deliberate scope decision
vector_query_adapter.py documents for pgvector.

token_cost is always 0.0: unlike litellm, which computes response_cost itself, the raw
anthropic SDK has no built-in pricing table to derive it from.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime, timezone

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step
from cascaid.ingestion.schema import CallEvent, NodeType


def _call_event(
    model: str, start_time: datetime, end_time: datetime, *, run_id: str, step: int, error: bool
) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario="production",
        step=step,
        caller=current_node.get(),
        callee=model,
        caller_type=NodeType.AGENT,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=(end_time - start_time).total_seconds() * 1000,
        error=error,
        retried=False,
        token_cost=0.0,
        occurred_at=datetime.now(timezone.utc),
    )


def _wrap_create(original: Callable, sink: Callable[[CallEvent], None]) -> Callable:
    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        run_id, step = current_run_id.get(), current_step.get()
        model = kwargs.get("model")
        start_time = datetime.now(timezone.utc)
        try:
            result = original(self, *args, **kwargs)
        except Exception:
            if run_id is not None and step is not None:
                sink(_call_event(model, start_time, datetime.now(timezone.utc), run_id=run_id, step=step, error=True))
            raise
        if run_id is not None and step is not None:
            sink(_call_event(model, start_time, datetime.now(timezone.utc), run_id=run_id, step=step, error=False))
        return result

    patched.__cascaid_instrumented__ = True
    return patched


def _wrap_async_create(original: Callable, sink: Callable[[CallEvent], None]) -> Callable:
    @functools.wraps(original)
    async def patched(self, *args, **kwargs):
        run_id, step = current_run_id.get(), current_step.get()
        model = kwargs.get("model")
        start_time = datetime.now(timezone.utc)
        try:
            result = await original(self, *args, **kwargs)
        except Exception:
            if run_id is not None and step is not None:
                sink(_call_event(model, start_time, datetime.now(timezone.utc), run_id=run_id, step=step, error=True))
            raise
        if run_id is not None and step is not None:
            sink(_call_event(model, start_time, datetime.now(timezone.utc), run_id=run_id, step=step, error=False))
        return result

    patched.__cascaid_instrumented__ = True
    return patched


def instrument_anthropic(sink: Callable[[CallEvent], None]) -> None:
    """Patches anthropic.resources.messages.Messages.create and AsyncMessages.create so
    every direct client.messages.create() call in the customer's own code is observed
    with zero call-site changes. Each currently-installed method is wrapped at most once
    (marked via __cascaid_instrumented__ on the patched function itself) so a repeated
    bootstrap call can't stack duplicate instrumentation layers and double-sink events.

    Skips sinking (but never raises into the customer's call path) when
    current_run_id/current_step aren't set -- instrumentation hasn't reached this call
    for some reason, same guard as register_litellm_callbacks.
    """
    import anthropic.resources.messages as messages_module

    if not getattr(messages_module.Messages.create, "__cascaid_instrumented__", False):
        messages_module.Messages.create = _wrap_create(messages_module.Messages.create, sink)
    if not getattr(messages_module.AsyncMessages.create, "__cascaid_instrumented__", False):
        messages_module.AsyncMessages.create = _wrap_async_create(messages_module.AsyncMessages.create, sink)
