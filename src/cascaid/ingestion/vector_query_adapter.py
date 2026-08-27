"""Times a vector-store query at its call site and produces a CallEvent (PRD section
4.5, runtime seam). Covers pgvector, Pinecone, and Weaviate uniformly: none of them
expose a callback registry like litellm's, so this needs an explicit wrapper around the
query call rather than a passive hook -- and since the wrapper only measures elapsed
time and catches exceptions around an arbitrary block, it doesn't care which client
library is inside it.
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
