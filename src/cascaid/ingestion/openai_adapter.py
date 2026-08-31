"""Direct OpenAI SDK instrumentation (PRD section 4.5, runtime seam): for pipelines that
call openai.OpenAI()/AsyncOpenAI() directly rather than through litellm. See
docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md for why this shipped after
Anthropic, and for the dedup problem this module exists to solve.

Same shape as anthropic_adapter.py in every other respect (no callback registry on the
SDK, so this patches Completions.create/AsyncCompletions.create directly and sinks a
CallEvent inline, in the same call stack/coroutine as the original call -- no
metadata-snapshot workaround needed). The one real difference: litellm's OpenAI provider
path internally calls openai_client.chat.completions.with_raw_response.create(...) --
verified empirically (see ADR 0001 and litellm_adapter.py's inside_litellm_dispatch
docstring), with_raw_response is a @cached_property that looks up completions.create
dynamically, so it resolves through this module's own patch too. Without a dedup check,
a litellm.completion() call for an OpenAI model would produce two CallEvents for one
logical call: one from litellm_adapter's callback registry, one from here. This module
checks litellm_adapter.inside_litellm_dispatch and skips sinking (but still calls
through) when it's set, so litellm's own adapter is the one that records that call, and
a customer's own direct create() call elsewhere is unaffected (the flag is only true
during litellm's own dispatch window, not for the process's whole lifetime).

Streaming is out of scope for this pass, same as anthropic_adapter.py, and for the same
reason vector_query_adapter.py excludes pgvector -- a deliberate scope decision, not an
oversight. dedup coverage here also assumes litellm's real HTTP dispatch happens on the
same thread/task as the patched_completion/patched_acompletion call that sets the flag;
litellm's own module docstring notes a *streaming* sync completion() call is dispatched
via a background ThreadPoolExecutor that doesn't propagate contextvars -- since streaming
isn't wired up on either side of this dedup pair yet, that gap doesn't apply today, but
would need addressing before streaming support is added to either adapter.

token_cost is always 0.0, same reasoning as anthropic_adapter.py: the raw SDK has no
built-in pricing table to derive it from.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime, timezone

from cascaid.ingestion.litellm_adapter import inside_litellm_dispatch
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
        if inside_litellm_dispatch.get():
            return original(self, *args, **kwargs)

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
        if inside_litellm_dispatch.get():
            return await original(self, *args, **kwargs)

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


def instrument_openai(sink: Callable[[CallEvent], None]) -> None:
    """Patches openai.resources.chat.completions.Completions.create and
    AsyncCompletions.create so every direct client.chat.completions.create() call in the
    customer's own code is observed with zero call-site changes. Each currently-installed
    method is wrapped at most once (marked via __cascaid_instrumented__ on the patched
    function itself) so a repeated bootstrap call can't stack duplicate instrumentation
    layers and double-sink events.

    Skips sinking (but never raises into the customer's call path) when
    current_run_id/current_step aren't set, or when litellm_adapter.inside_litellm_dispatch
    is set (see module docstring) -- same guard shape as anthropic_adapter's
    instrument_anthropic, plus the dedup check.
    """
    import openai.resources.chat.completions as completions_module

    if not getattr(completions_module.Completions.create, "__cascaid_instrumented__", False):
        completions_module.Completions.create = _wrap_create(completions_module.Completions.create, sink)
    if not getattr(completions_module.AsyncCompletions.create, "__cascaid_instrumented__", False):
        completions_module.AsyncCompletions.create = _wrap_async_create(
            completions_module.AsyncCompletions.create, sink
        )
