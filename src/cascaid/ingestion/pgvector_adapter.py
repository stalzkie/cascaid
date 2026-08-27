"""Times a pgvector query at its call site and produces a CallEvent (PRD section 4.5,
runtime seam). pgvector has no callback registry like litellm's, so unlike the model-
endpoint adapter this needs an explicit wrapper around the query call.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from cascaid.ingestion.schema import CallEvent, NodeType


class VectorQueryTracker:
    def __init__(self):
        self.event: CallEvent | None = None


@contextmanager
def observe_vector_query(
    caller: str,
    callee: str,
    *,
    run_id: str,
    step: int,
    scenario: str = "production",
    caller_type: NodeType = NodeType.TOOL,
):
    tracker = VectorQueryTracker()
    start = time.perf_counter()
    error = False
    try:
        yield tracker
    except Exception:
        error = True
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        tracker.event = CallEvent(
            run_id=run_id,
            scenario=scenario,
            step=step,
            caller=caller,
            callee=callee,
            caller_type=caller_type,
            callee_type=NodeType.VECTOR_STORE,
            latency_ms=latency_ms,
            error=error,
            retried=False,
            token_cost=0.0,
        )
