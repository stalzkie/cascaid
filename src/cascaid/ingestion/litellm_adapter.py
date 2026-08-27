"""Converts a LiteLLM success-callback invocation into a CallEvent (PRD section 4.5,
runtime seam): reads existing LiteLLM callback data, no new instrumentation at call
sites. Caller attribution comes from runtime_context.current_node, set by the LangGraph
adapter around each node's execution.
"""

from __future__ import annotations

from datetime import datetime

from cascaid.ingestion.runtime_context import current_node
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
