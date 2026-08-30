"""Times a vector-store query at its call site and produces a CallEvent (PRD section
4.5, runtime seam). Covers pgvector, Pinecone, and Weaviate uniformly: none of them
expose a callback registry like litellm's, so this needs an explicit wrapper around the
query call rather than a passive hook -- and since the wrapper only measures elapsed
time and catches exceptions around an arbitrary block, it doesn't care which client
library is inside it.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step
from cascaid.ingestion.schema import CallEvent, NodeType


class VectorQueryTracker:
    def __init__(self):
        self.event: CallEvent | None = None


def _build_vector_call_event(
    caller: str, callee: str, run_id: str, step: int, scenario: str, caller_type: NodeType, error: bool, start: float
) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario=scenario,
        step=step,
        caller=caller,
        callee=callee,
        caller_type=caller_type,
        callee_type=NodeType.VECTOR_STORE,
        latency_ms=(time.perf_counter() - start) * 1000,
        error=error,
        retried=False,
        token_cost=0.0,
        occurred_at=datetime.now(timezone.utc),
    )


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
        tracker.event = _build_vector_call_event(caller, callee, run_id, step, scenario, caller_type, error, start)


@asynccontextmanager
async def observe_vector_query_async(
    caller: str,
    callee: str,
    *,
    run_id: str,
    step: int,
    scenario: str = "production",
    caller_type: NodeType = NodeType.TOOL,
):
    """Async twin of observe_vector_query -- needed for vector DB clients that
    expose an async query surface (e.g. Weaviate's _QueryCollectionAsync, see
    docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md)."""
    tracker = VectorQueryTracker()
    start = time.perf_counter()
    error = False
    try:
        yield tracker
    except Exception:
        error = True
        raise
    finally:
        tracker.event = _build_vector_call_event(caller, callee, run_id, step, scenario, caller_type, error, start)


# Auto-patch surface (PRD 4.5, runtime seam): verified via introspection against
# the installed pinecone/weaviate-client packages, not assumed from memory --
# neither vendor exposes one "query method", each has several. Patching only the
# obvious one (Index.query / Collection.query.near_vector) would silently
# under-count real retrieval activity, understating vector-store load to the GNN
# -- an accuracy problem, not just a coverage gap. `fetch`-family lookups are
# included too even though they're not similarity search, since they still hit
# the vector store and still carry real latency/error signal.
#
# Re-verified 2026-08-30 against pinecone==9.1.0 (docs/Production_Readiness_
# and_Pipeline_Compatibility_Assessment.md): search_records (an alias for
# search) and fetch_by_metadata (a metadata-filtered fetch) were added to
# Index since this list was last checked and were missing here -- the exact
# under-counting failure mode described above, now closed. Re-verify this
# list whenever pinecone's pinned version bumps.
PINECONE_QUERY_METHODS = ["query", "query_namespaces", "search", "search_records", "fetch", "fetch_by_metadata"]
WEAVIATE_QUERY_METHODS = [
    "near_vector",
    "near_text",
    "near_object",
    "near_image",
    "near_media",
    "hybrid",
    "bm25",
    "fetch_objects",
    "fetch_object_by_id",
    "fetch_objects_by_ids",
]


def _wrap_vector_method(vendor_label: str, original: Callable, sink_cell: list):
    """sink_cell is a 1-element list so register_*_callbacks can retarget the sink
    on a repeat call without re-patching (same pattern as instrument_langgraph's
    module-level sink) -- each patched method gets its own cell, no shared mutable
    state across different methods or different query calls.

    Dispatches to an async wrapper when `original` is a coroutine function (e.g.
    Weaviate's _QueryCollectionAsync) -- a sync wrapper around an async method
    would return an unawaited coroutine and exit its `with` block before the
    real query ever ran, the same class of bug fixed in langgraph_adapter.py's
    _wrap_invoke/_wrap_node_fn (see
    docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md)."""
    if asyncio.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapped_async(self, *args, **kwargs):
            run_id, step = current_run_id.get(), current_step.get()
            if run_id is None or step is None:
                return await original(self, *args, **kwargs)

            callee = getattr(self, "name", vendor_label)
            caller = current_node.get() or "unknown"
            try:
                async with observe_vector_query_async(caller, callee, run_id=run_id, step=step) as tracker:
                    result = await original(self, *args, **kwargs)
                return result
            finally:
                if tracker.event is not None:
                    sink_cell[0](tracker.event)

        return wrapped_async

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        run_id, step = current_run_id.get(), current_step.get()
        if run_id is None or step is None:
            return original(self, *args, **kwargs)

        callee = getattr(self, "name", vendor_label)
        caller = current_node.get() or "unknown"
        try:
            with observe_vector_query(caller, callee, run_id=run_id, step=step) as tracker:
                result = original(self, *args, **kwargs)
            return result
        finally:
            if tracker.event is not None:
                sink_cell[0](tracker.event)

    return wrapped


def _patch_methods(cls, method_names: list[str], vendor_label: str, sink: Callable[[CallEvent], None]) -> list:
    sink_cell = [sink]
    for name in method_names:
        original = getattr(cls, name)
        setattr(cls, name, _wrap_vector_method(vendor_label, original, sink_cell))
    return sink_cell


_patched_pinecone_sink: list[Callable[[CallEvent], None]] | None = None
_patched_weaviate_sink: list[Callable[[CallEvent], None]] | None = None
_patched_weaviate_async_sink: list[Callable[[CallEvent], None]] | None = None


def register_pinecone_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches every Pinecone Index query-shaped method (PINECONE_QUERY_METHODS)
    to auto-record a CallEvent via the existing observe_vector_query converter,
    reading run_id/step/caller from the runtime context the same way
    register_litellm_callbacks does. No-ops (calls through, doesn't raise) if
    that context isn't set yet."""
    global _patched_pinecone_sink
    from pinecone import Index

    if _patched_pinecone_sink is not None:
        _patched_pinecone_sink[0] = sink
        return
    _patched_pinecone_sink = _patch_methods(Index, PINECONE_QUERY_METHODS, "pinecone", sink)


def register_weaviate_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches every Weaviate query-shaped method on both _QueryCollection
    (sync client) and _QueryCollectionAsync (weaviate's separate async client,
    e.g. WeaviateAsyncClient/use_async_with_weaviate_cloud -- same method
    names, a fully distinct class, previously unpatched: see
    docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md) the
    same way register_pinecone_callbacks does."""
    global _patched_weaviate_sink, _patched_weaviate_async_sink
    from weaviate.collections.collection.async_ import _QueryCollectionAsync
    from weaviate.collections.collection.sync import _QueryCollection

    if _patched_weaviate_sink is not None:
        _patched_weaviate_sink[0] = sink
    else:
        _patched_weaviate_sink = _patch_methods(_QueryCollection, WEAVIATE_QUERY_METHODS, "weaviate", sink)

    if _patched_weaviate_async_sink is not None:
        _patched_weaviate_async_sink[0] = sink
    else:
        _patched_weaviate_async_sink = _patch_methods(_QueryCollectionAsync, WEAVIATE_QUERY_METHODS, "weaviate", sink)
