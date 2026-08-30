"""Converts a LiteLLM success-callback invocation into a CallEvent (PRD section 4.5,
runtime seam): reads existing LiteLLM callback data, no new instrumentation at call
sites. Caller attribution comes from runtime_context.current_node, set by the LangGraph
adapter around each node's execution.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step
from cascaid.ingestion.schema import CallEvent, NodeType


def litellm_success_to_call_event(
    kwargs: dict,
    completion_response,
    start_time: datetime,
    end_time: datetime,
    *,
    run_id: str,
    step: int,
    scenario: str = "production",
    caller_type: NodeType = NodeType.AGENT,
) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario=scenario,
        step=step,
        caller=current_node.get(),
        callee=kwargs["model"],
        caller_type=caller_type,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=(end_time - start_time).total_seconds() * 1000,
        error=False,
        retried=False,
        token_cost=kwargs.get("response_cost") or 0.0,
    )


def litellm_failure_to_call_event(
    kwargs: dict,
    completion_response,
    start_time: datetime,
    end_time: datetime,
    *,
    run_id: str,
    step: int,
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
        caller=current_node.get(),
        callee=kwargs["model"],
        caller_type=caller_type,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=(end_time - start_time).total_seconds() * 1000,
        error=True,
        retried=False,
        token_cost=kwargs.get("response_cost") or 0.0,
    )


def register_litellm_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Wires the converters above into litellm's own callback registry (PRD 4.5,
    runtime seam) -- appends, never replaces, so this composes with a customer's
    existing Langfuse/LangSmith/Phoenix callback instead of clobbering it.

    Requires current_run_id/current_step to already be set (by the LangGraph
    invocation-boundary adapter) -- if instrumentation hasn't reached this call for
    some reason, the hook no-ops rather than raising into the customer's live call
    path or fabricating a run_id/step that would misrepresent the pipeline."""
    import litellm

    def _on_success(kwargs, completion_response, start_time, end_time):
        run_id, step = current_run_id.get(), current_step.get()
        if run_id is None or step is None:
            return
        sink(litellm_success_to_call_event(kwargs, completion_response, start_time, end_time, run_id=run_id, step=step))

    def _on_failure(kwargs, completion_response, start_time, end_time):
        run_id, step = current_run_id.get(), current_step.get()
        if run_id is None or step is None:
            return
        sink(litellm_failure_to_call_event(kwargs, completion_response, start_time, end_time, run_id=run_id, step=step))

    litellm.success_callback.append(_on_success)
    litellm.failure_callback.append(_on_failure)
