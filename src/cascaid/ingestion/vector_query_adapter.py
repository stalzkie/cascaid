"""Times a vector-store query at its call site and produces a CallEvent (PRD section
4.5, runtime seam). Covers pgvector, Pinecone, Weaviate, Chroma, Qdrant, Milvus, and
LanceDB uniformly: none of them expose a callback registry like litellm's, so this needs
an explicit wrapper around the query call rather than a passive hook -- and since the
wrapper only measures elapsed time and catches exceptions around an arbitrary block, it
doesn't care which client library is inside it.

callee (the vector store's identity in a CallEvent) is derived differently across
vendors: Pinecone/Weaviate/Chroma's query-shaped methods are called on a per-collection
object that exposes its own `.name` (the default `getattr(self, "name", vendor_label)`
in _wrap_vector_method covers these). Qdrant and Milvus's clients are NOT
per-collection -- one QdrantClient/MilvusClient instance is called with
`collection_name=...` as a kwarg on every query -- so those register_*_callbacks calls
pass an explicit `callee_from` extractor instead.
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

# Verified 2026-08-31 via introspection against the installed chromadb==1.1.1 --
# chromadb.api.models.Collection.Collection (and its async twin, AsyncCollection,
# same method names). query/get/peek/search are read-shaped and hit the store; count()
# is included for the same reason PINECONE_QUERY_METHODS includes fetch-family lookups
# even though it's not similarity search.
CHROMA_QUERY_METHODS = ["query", "get", "peek", "search", "count"]

# Verified 2026-08-31 against qdrant-client==1.19.0 -- this version has no `search`/
# `recommend` methods at all (fully replaced by query_points/query_points_groups/
# query_batch_points in this API generation); QdrantClient is NOT per-collection, every
# call takes collection_name as a kwarg, so register_qdrant_callbacks passes an explicit
# callee_from extractor rather than relying on _wrap_vector_method's `self.name` default.
QDRANT_QUERY_METHODS = [
    "query_points",
    "query_points_groups",
    "query_batch_points",
    "scroll",
    "retrieve",
    "count",
    "search_matrix_offsets",
    "search_matrix_pairs",
]

# Verified 2026-08-31 against pymilvus==3.0.1 -- MilvusClient/AsyncMilvusClient are also
# not per-collection (collection_name is a kwarg, same situation as Qdrant).
# query_iterator/search_iterator exist on the sync client only (no async equivalent in
# this version).
MILVUS_QUERY_METHODS = ["search", "hybrid_search", "query", "query_iterator", "search_iterator", "get"]
MILVUS_ASYNC_QUERY_METHODS = ["search", "hybrid_search", "query", "get"]


def _default_callee(vendor_label: str) -> Callable[[object, tuple, dict], str]:
    return lambda self, args, kwargs: getattr(self, "name", vendor_label)


def _wrap_vector_method(
    vendor_label: str,
    original: Callable,
    sink_cell: list,
    callee_from: Callable[[object, tuple, dict], str] | None = None,
):
    """sink_cell is a 1-element list so register_*_callbacks can retarget the sink
    on a repeat call without re-patching (same pattern as instrument_langgraph's
    module-level sink) -- each patched method gets its own cell, no shared mutable
    state across different methods or different query calls.

    callee_from defaults to reading `self.name` (Pinecone/Weaviate/Chroma's
    per-collection objects); pass an explicit extractor for a vendor whose client
    isn't per-collection (see module docstring).

    Dispatches to an async wrapper when `original` is a coroutine function (e.g.
    Weaviate's _QueryCollectionAsync) -- a sync wrapper around an async method
    would return an unawaited coroutine and exit its `with` block before the
    real query ever ran, the same class of bug fixed in langgraph_adapter.py's
    _wrap_invoke/_wrap_node_fn (see
    docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md)."""
    callee_from = callee_from or _default_callee(vendor_label)

    if asyncio.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapped_async(self, *args, **kwargs):
            run_id, step = current_run_id.get(), current_step.get()
            if run_id is None or step is None:
                return await original(self, *args, **kwargs)

            callee = callee_from(self, args, kwargs)
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

        callee = callee_from(self, args, kwargs)
        caller = current_node.get() or "unknown"
        try:
            with observe_vector_query(caller, callee, run_id=run_id, step=step) as tracker:
                result = original(self, *args, **kwargs)
            return result
        finally:
            if tracker.event is not None:
                sink_cell[0](tracker.event)

    return wrapped


def _patch_methods(
    cls,
    method_names: list[str],
    vendor_label: str,
    sink: Callable[[CallEvent], None],
    callee_from: Callable[[object, tuple, dict], str] | None = None,
) -> list:
    sink_cell = [sink]
    for name in method_names:
        original = getattr(cls, name)
        setattr(cls, name, _wrap_vector_method(vendor_label, original, sink_cell, callee_from))
    return sink_cell


def _collection_name_kwarg(vendor_label: str) -> Callable[[object, tuple, dict], str]:
    """Qdrant/Milvus clients aren't per-collection -- collection_name is the first
    positional-or-keyword parameter on every query-shaped method (verified via
    introspection), so it may arrive either way."""

    def extract(self, args: tuple, kwargs: dict) -> str:
        return kwargs.get("collection_name") or (args[0] if args else vendor_label)

    return extract


_patched_pinecone_sink: list[Callable[[CallEvent], None]] | None = None
_patched_weaviate_sink: list[Callable[[CallEvent], None]] | None = None
_patched_weaviate_async_sink: list[Callable[[CallEvent], None]] | None = None
_patched_chroma_sink: list[Callable[[CallEvent], None]] | None = None
_patched_chroma_async_sink: list[Callable[[CallEvent], None]] | None = None
_patched_qdrant_sink: list[Callable[[CallEvent], None]] | None = None
_patched_qdrant_async_sink: list[Callable[[CallEvent], None]] | None = None
_patched_milvus_sink: list[Callable[[CallEvent], None]] | None = None
_patched_milvus_async_sink: list[Callable[[CallEvent], None]] | None = None
_patched_lancedb_sinks: list[list[Callable[[CallEvent], None]]] | None = None


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


def register_chroma_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches every Chroma query-shaped method (CHROMA_QUERY_METHODS) on both
    Collection (sync) and AsyncCollection (chromadb's separate async client) --
    both expose `.name`, so this uses _wrap_vector_method's default callee_from,
    same as register_pinecone_callbacks."""
    global _patched_chroma_sink, _patched_chroma_async_sink
    from chromadb.api.models.AsyncCollection import AsyncCollection
    from chromadb.api.models.Collection import Collection

    if _patched_chroma_sink is not None:
        _patched_chroma_sink[0] = sink
    else:
        _patched_chroma_sink = _patch_methods(Collection, CHROMA_QUERY_METHODS, "chroma", sink)

    if _patched_chroma_async_sink is not None:
        _patched_chroma_async_sink[0] = sink
    else:
        _patched_chroma_async_sink = _patch_methods(AsyncCollection, CHROMA_QUERY_METHODS, "chroma", sink)


def register_qdrant_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches every Qdrant query-shaped method (QDRANT_QUERY_METHODS) on both
    QdrantClient (sync) and AsyncQdrantClient. Unlike Pinecone/Weaviate/Chroma,
    QdrantClient isn't per-collection -- collection_name is a call-time argument, so
    callee comes from _collection_name_kwarg instead of `self.name`."""
    global _patched_qdrant_sink, _patched_qdrant_async_sink
    from qdrant_client import AsyncQdrantClient, QdrantClient

    callee_from = _collection_name_kwarg("qdrant")

    if _patched_qdrant_sink is not None:
        _patched_qdrant_sink[0] = sink
    else:
        _patched_qdrant_sink = _patch_methods(QdrantClient, QDRANT_QUERY_METHODS, "qdrant", sink, callee_from)

    if _patched_qdrant_async_sink is not None:
        _patched_qdrant_async_sink[0] = sink
    else:
        _patched_qdrant_async_sink = _patch_methods(
            AsyncQdrantClient, QDRANT_QUERY_METHODS, "qdrant", sink, callee_from
        )


def register_milvus_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches every Milvus query-shaped method on both MilvusClient (sync,
    MILVUS_QUERY_METHODS) and AsyncMilvusClient (async, MILVUS_ASYNC_QUERY_METHODS --
    a narrower list, this SDK version has no async query_iterator/search_iterator).
    Same not-per-collection situation as Qdrant."""
    global _patched_milvus_sink, _patched_milvus_async_sink
    from pymilvus import AsyncMilvusClient, MilvusClient

    callee_from = _collection_name_kwarg("milvus")

    if _patched_milvus_sink is not None:
        _patched_milvus_sink[0] = sink
    else:
        _patched_milvus_sink = _patch_methods(MilvusClient, MILVUS_QUERY_METHODS, "milvus", sink, callee_from)

    if _patched_milvus_async_sink is not None:
        _patched_milvus_async_sink[0] = sink
    else:
        _patched_milvus_async_sink = _patch_methods(
            AsyncMilvusClient, MILVUS_ASYNC_QUERY_METHODS, "milvus", sink, callee_from
        )


def register_lancedb_callbacks(sink: Callable[[CallEvent], None]) -> None:
    """Patches LanceQueryBuilder.to_arrow on LanceDB's concrete sync query-builder
    subclasses (LanceVectorQueryBuilder, LanceFtsQueryBuilder, LanceHybridQueryBuilder,
    LanceEmptyQueryBuilder).

    LanceDB's query API is lazy, unlike every other vendor here: `table.search(...)`
    just builds a LanceQueryBuilder and does not touch the store -- the real query
    executes only when a terminal method (to_pandas/to_list/to_polars/to_pydantic/
    to_arrow) is called on that builder (verified via introspection: all four of those
    delegate to `self.to_arrow(...)` as their one common execution primitive, so
    patching `to_arrow` alone covers every terminal method with exactly one CallEvent
    per logical query, however the customer chose to materialize the result).
    `to_arrow` is `@abstractmethod` on the base LanceQueryBuilder and each concrete
    subclass supplies its own implementation, so the base class must NOT be patched
    (Python's MRO would never reach it) -- each subclass needs patching individually.

    Deliberately out of scope for this pass (not oversight): LanceDB's separate async
    query classes (AsyncQuery/AsyncVectorQuery/etc. in lancedb.query), and
    LanceTakeQueryBuilder (`.take_offsets()`/`.take_row_ids()`, a distinct, simpler
    builder hierarchy not descended from LanceQueryBuilder). callee falls back to the
    "lancedb" vendor label -- the query builder only reaches the underlying table via a
    private `_inner` reference, too fragile to chase down here."""
    global _patched_lancedb_sinks
    from lancedb.query import (
        LanceEmptyQueryBuilder,
        LanceFtsQueryBuilder,
        LanceHybridQueryBuilder,
        LanceVectorQueryBuilder,
    )

    builder_classes = [LanceVectorQueryBuilder, LanceFtsQueryBuilder, LanceHybridQueryBuilder, LanceEmptyQueryBuilder]

    if _patched_lancedb_sinks is not None:
        for sink_cell in _patched_lancedb_sinks:
            sink_cell[0] = sink
        return
    _patched_lancedb_sinks = [_patch_methods(cls, ["to_arrow"], "lancedb", sink) for cls in builder_classes]
