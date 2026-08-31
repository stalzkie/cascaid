"""Direct Gemini SDK instrumentation (PRD section 4.5, runtime seam): for pipelines that
call google.genai.Client() directly rather than through litellm. See
docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md for the sequencing this
followed (Anthropic first, OpenAI needed a dedup mechanism, Gemini shipped alongside
Anthropic once confirmed to have the same no-risk situation).

Confirmed empirically (grep against litellm/llms/gemini/ and litellm/llms/vertex_ai/,
recorded in ADR 0001): litellm's Gemini/Vertex path never imports
google.generativeai/google.genai anywhere, so -- unlike OpenAI -- there's no internal
client litellm creates that this adapter's patch could double-count. Same shape as
anthropic_adapter.py otherwise: no callback registry on the SDK, so this patches
Models.generate_content/AsyncModels.generate_content directly (verified via
introspection against the installed google-genai package, not assumed from memory --
`genai.Client(...).models` is a `google.genai.models.Models` instance,
`client.aio.models` is a `google.genai.models.AsyncModels` instance, both exposing
`generate_content` as a keyword-only-model method distinct from the streaming
`generate_content_stream`) and sinks a CallEvent inline, in the same call
stack/coroutine as the original call -- no metadata-snapshot workaround needed.

generate_content_stream (streaming) is out of scope for this pass, same deliberate
scope decision as anthropic_adapter.py's client.messages.stream() and
vector_query_adapter.py's pgvector exclusion.

token_cost is always 0.0, same reasoning as anthropic_adapter.py: the raw SDK has no
built-in pricing table to derive it from.
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


def _wrap_generate_content(original: Callable, sink: Callable[[CallEvent], None]) -> Callable:
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


def _wrap_async_generate_content(original: Callable, sink: Callable[[CallEvent], None]) -> Callable:
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


def instrument_gemini(sink: Callable[[CallEvent], None]) -> None:
    """Patches google.genai.models.Models.generate_content and
    AsyncModels.generate_content so every direct client.models.generate_content()/
    client.aio.models.generate_content() call in the customer's own code is observed
    with zero call-site changes. Each currently-installed method is wrapped at most once
    (marked via __cascaid_instrumented__ on the patched function itself) so a repeated
    bootstrap call can't stack duplicate instrumentation layers and double-sink events.

    Skips sinking (but never raises into the customer's call path) when
    current_run_id/current_step aren't set -- instrumentation hasn't reached this call
    for some reason, same guard as instrument_anthropic/instrument_openai.
    """
    import google.genai.models as models_module

    if not getattr(models_module.Models.generate_content, "__cascaid_instrumented__", False):
        models_module.Models.generate_content = _wrap_generate_content(models_module.Models.generate_content, sink)
    if not getattr(models_module.AsyncModels.generate_content, "__cascaid_instrumented__", False):
        models_module.AsyncModels.generate_content = _wrap_async_generate_content(
            models_module.AsyncModels.generate_content, sink
        )
