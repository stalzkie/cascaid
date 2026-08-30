"""Converts a LiteLLM callback invocation into a CallEvent (PRD section 4.5,
runtime seam): reads existing LiteLLM callback data, no new instrumentation at
call sites.

Two dispatch mechanisms are used deliberately, not one:

- Sync `litellm.completion()` calls: the legacy `litellm.success_callback`/
  `failure_callback` lists.
- Async `litellm.acompletion()` calls: `litellm.callbacks` (a CustomLogger),
  because the legacy lists don't fire for acompletion() at all -- verified
  empirically (see docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md).
  litellm also defers this async dispatch to run detached from the coroutine
  that made the call, often well after it returns.

A single unified CustomLogger-only design for both was tried first and
reverted: it measurably slowed down litellm's own callback dispatch under
real load (observed making unrelated tests' short timeouts flaky), for no
benefit over the sync path the legacy lists already handled correctly.

Both mechanisms read run_id/step/caller from a `metadata` snapshot taken
*before* dispatch (see _snapshot_context_into_metadata), not ambient
contextvars at logging time -- current_run_id/current_step/current_node can
be stale or altogether unset by the time a callback actually fires. This
isn't only true for the async path: a *streaming* sync completion() call
(stream=True) is also dispatched via a background ThreadPoolExecutor
(verified empirically), so ambient contextvar reads would silently drop
every streaming CallEvent even on the "synchronous" path. litellm also fires
one success callback per streaming chunk plus a final aggregated one -- only
the final call (kwargs["complete_streaming_response"] is set) is sinked, so
streaming still produces exactly one CallEvent per logical LLM call.
register_litellm_callbacks patches litellm.completion/acompletion themselves
to take that metadata snapshot automatically.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime, timezone

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step
from cascaid.ingestion.schema import CallEvent, NodeType

_METADATA_RUN_ID = "cascaid_run_id"
_METADATA_STEP = "cascaid_step"
_METADATA_NODE = "cascaid_node"


def _cascaid_metadata(kwargs: dict) -> dict:
    return kwargs.get("litellm_params", {}).get("metadata") or {}


def litellm_success_to_call_event(
    kwargs: dict,
    completion_response,
    start_time: datetime,
    end_time: datetime,
    *,
    run_id: str,
    step: int,
    caller: str | None = None,
    scenario: str = "production",
    caller_type: NodeType = NodeType.AGENT,
) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario=scenario,
        step=step,
        caller=caller if caller is not None else current_node.get(),
        callee=kwargs["model"],
        caller_type=caller_type,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=(end_time - start_time).total_seconds() * 1000,
        error=False,
        retried=False,
        token_cost=kwargs.get("response_cost") or 0.0,
        # datetime.now(), not litellm's own start_time/end_time -- those are naive
        # local-clock timestamps with no guaranteed timezone, and occurred_at needs
        # to compare against IncidentLabel.occurred_at (stored tz-aware UTC).
        occurred_at=datetime.now(timezone.utc),
    )


def litellm_failure_to_call_event(
    kwargs: dict,
    completion_response,
    start_time: datetime,
    end_time: datetime,
    *,
    run_id: str,
    step: int,
    caller: str | None = None,
    scenario: str = "production",
    caller_type: NodeType = NodeType.AGENT,
) -> CallEvent:
    # litellm's failure callback fires once per final outcome; it doesn't expose how
    # many attempts were retried internally before failing, so retried is always False
    # here -- real retry tracking needs call-attempt correlation across time, out of
    # scope for a single-event converter.
    return CallEvent(
        run_id=run_id,
        scenario=scenario,
        step=step,
        caller=caller if caller is not None else current_node.get(),
        callee=kwargs["model"],
        caller_type=caller_type,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=(end_time - start_time).total_seconds() * 1000,
        error=True,
        retried=False,
        token_cost=kwargs.get("response_cost") or 0.0,
        occurred_at=datetime.now(timezone.utc),
    )


def _build_async_cascaid_logger(sink: Callable[[CallEvent], None]):
    # Async-only CustomLogger: the sync log_success_event/log_failure_event
    # hooks are deliberately left at CustomLogger's own no-op default, since
    # the sync path is already covered (faster and proven) by the legacy
    # success_callback/failure_callback lists below -- overriding them here
    # too would just make litellm invoke this logger twice for every sync call.
    from litellm.integrations.custom_logger import CustomLogger

    class _CascaidAsyncLoggingCallback(CustomLogger):
        async def _dispatch(self, converter, kwargs, completion_response, start_time, end_time):
            metadata = _cascaid_metadata(kwargs)
            run_id, step = metadata.get(_METADATA_RUN_ID), metadata.get(_METADATA_STEP)
            if run_id is None or step is None:
                return
            sink(
                converter(
                    kwargs,
                    completion_response,
                    start_time,
                    end_time,
                    run_id=run_id,
                    step=step,
                    caller=metadata.get(_METADATA_NODE),
                )
            )

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            await self._dispatch(litellm_success_to_call_event, kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            await self._dispatch(litellm_failure_to_call_event, kwargs, response_obj, start_time, end_time)

    return _CascaidAsyncLoggingCallback()


_patched_completion = False


def _snapshot_context_into_metadata(kwargs: dict) -> dict:
    """Captures current_run_id/current_step/current_node into the call's own
    `metadata` kwarg *before* the request is dispatched -- at this point the
    contextvars are guaranteed valid (we're inside the customer's node
    execution), unlike at logging-callback time (see module docstring). Merges
    with, never overwrites, any metadata the customer's own code already set."""
    run_id, step, node = current_run_id.get(), current_step.get(), current_node.get()
    if run_id is None or step is None:
        return kwargs
    metadata = dict(kwargs.get("metadata") or {})
    metadata.setdefault(_METADATA_RUN_ID, run_id)
    metadata.setdefault(_METADATA_STEP, step)
    metadata.setdefault(_METADATA_NODE, node)
    kwargs["metadata"] = metadata
    return kwargs


def _patch_completion_functions() -> None:
    """Patches litellm.completion/acompletion themselves (once per process) so
    the metadata snapshot above happens automatically -- the customer's code
    never has to pass `metadata=` itself."""
    global _patched_completion
    if _patched_completion:
        return
    _patched_completion = True

    import litellm

    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    @functools.wraps(original_completion)
    def patched_completion(*args, **kwargs):
        return original_completion(*args, **_snapshot_context_into_metadata(kwargs))

    @functools.wraps(original_acompletion)
    async def patched_acompletion(*args, **kwargs):
        return await original_acompletion(*args, **_snapshot_context_into_metadata(kwargs))

    litellm.completion = patched_completion
    litellm.acompletion = patched_acompletion


def register_litellm_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Wires the converters above into litellm's own callback registries (PRD
    4.5, runtime seam) -- appends, never replaces, so this composes with a
    customer's existing Langfuse/LangSmith/Phoenix callback instead of
    clobbering it. See the module docstring for why sync and async dispatch
    use two different mechanisms.

    Requires the metadata snapshot below to have been captured -- if
    instrumentation hasn't reached this call for some reason, the hook no-ops
    rather than raising into the customer's live call path or fabricating a
    run_id/step that would misrepresent the pipeline.

    Reads run_id/step/caller from the metadata snapshot (see
    _snapshot_context_into_metadata), not ambient contextvars, even on this
    "fast, synchronous" sync path: a *streaming* sync completion() call
    (stream=True) is dispatched by litellm via a background
    ThreadPoolExecutor -- verified empirically, same contextvar-loss failure
    mode as the async/thread cases documented in the module docstring -- so
    ambient contextvar reads would silently drop every streaming CallEvent.
    litellm also fires one success callback per streaming chunk plus a final
    aggregated one; only the final call (kwargs["complete_streaming_response"]
    is set) is sinked, so one logical LLM call still produces exactly one
    CallEvent instead of a run of near-duplicate, degenerate-latency ones."""
    import litellm

    def _on_success(kwargs, completion_response, start_time, end_time):
        if kwargs.get("stream") and kwargs.get("complete_streaming_response") is None:
            return  # mid-stream chunk, not litellm's final aggregated callback
        metadata = _cascaid_metadata(kwargs)
        run_id, step = metadata.get(_METADATA_RUN_ID), metadata.get(_METADATA_STEP)
        if run_id is None or step is None:
            return
        sink(
            litellm_success_to_call_event(
                kwargs,
                completion_response,
                start_time,
                end_time,
                run_id=run_id,
                step=step,
                caller=metadata.get(_METADATA_NODE),
            )
        )

    def _on_failure(kwargs, completion_response, start_time, end_time):
        metadata = _cascaid_metadata(kwargs)
        run_id, step = metadata.get(_METADATA_RUN_ID), metadata.get(_METADATA_STEP)
        if run_id is None or step is None:
            return
        sink(
            litellm_failure_to_call_event(
                kwargs,
                completion_response,
                start_time,
                end_time,
                run_id=run_id,
                step=step,
                caller=metadata.get(_METADATA_NODE),
            )
        )

    litellm.success_callback.append(_on_success)
    litellm.failure_callback.append(_on_failure)
    litellm.callbacks.append(_build_async_cascaid_logger(sink))
    _patch_completion_functions()
