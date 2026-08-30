import time
from datetime import timezone

import pytest

import cascaid.ingestion.vector_query_adapter as vector_query_adapter
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType
from cascaid.ingestion.vector_query_adapter import (
    PINECONE_QUERY_METHODS,
    WEAVIATE_QUERY_METHODS,
    observe_vector_query,
    register_pinecone_callbacks,
    register_weaviate_callbacks,
)


@pytest.fixture(autouse=True)
def _reset_patch_state():
    # register_*_callbacks patches once per process by design (matches
    # instrument_langgraph/register_litellm_callbacks) -- these tests reassign
    # Index.query/etc. directly per test to install a fresh stand-in, which needs
    # a fresh (not idempotent-skipped) patch each time to actually wrap it.
    vector_query_adapter._patched_pinecone_sink = None
    vector_query_adapter._patched_weaviate_sink = None
    yield
    vector_query_adapter._patched_pinecone_sink = None
    vector_query_adapter._patched_weaviate_sink = None


def test_observe_vector_query_records_success():
    with observe_vector_query("retriever_tool", "vector_store", run_id="run-1", step=2) as tracker:
        time.sleep(0.01)

    event = tracker.event
    assert event.run_id == "run-1"
    assert event.step == 2
    assert event.caller == "retriever_tool"
    assert event.caller_type == NodeType.TOOL
    assert event.callee == "vector_store"
    assert event.callee_type == NodeType.VECTOR_STORE
    assert event.error is False
    assert event.retried is False
    assert event.token_cost == 0.0
    assert event.latency_ms >= 10
    assert event.occurred_at is not None
    assert event.occurred_at.tzinfo == timezone.utc


def test_observe_vector_query_records_error_and_reraises():
    tracker_ref = {}

    with pytest.raises(RuntimeError, match="connection lost"):
        with observe_vector_query("retriever_tool", "vector_store", run_id="run-1", step=2) as tracker:
            tracker_ref["tracker"] = tracker
            raise RuntimeError("connection lost")

    event = tracker_ref["tracker"].event
    assert event.error is True


# --- register_pinecone_callbacks / register_weaviate_callbacks ---
#
# Neither vendor SDK offers an offline/mock dispatch mode the way LiteLLM's
# mock_response does (no local emulator, no built-in test double) -- these tests
# prove the generic wrapping mechanism against the *real* classes/method names
# (verified via introspection against the installed pinecone/weaviate-client
# packages), with a controlled stand-in substituted only at the innermost
# "network call" layer, the one thing that genuinely can't be exercised without
# a live backend.


def test_register_pinecone_callbacks_patches_every_query_method():
    from pinecone import Index

    originals = {name: getattr(Index, name) for name in PINECONE_QUERY_METHODS}
    try:
        register_pinecone_callbacks(sink=lambda event: None)
        for name in PINECONE_QUERY_METHODS:
            assert getattr(Index, name) is not originals[name]
    finally:
        for name, original in originals.items():
            setattr(Index, name, original)


def test_pinecone_query_methods_covers_search_records_and_fetch_by_metadata():
    # Regression test: pinecone>=9.1.0 added Index.search_records (an alias
    # for search -- confirmed via Index.search_records.__doc__) and
    # Index.fetch_by_metadata (a metadata-filtered fetch, same shape as the
    # already-covered `fetch`) since this list was last verified against the
    # real installed package. Both are retrieval/fetch-shaped and were
    # missing -- silently under-counting real retrieval activity, exactly
    # the failure mode this module's own docstring warns against. See
    # docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md.
    from pinecone import Index

    assert hasattr(Index, "search_records"), "pinecone's Index API changed -- re-verify PINECONE_QUERY_METHODS"
    assert hasattr(Index, "fetch_by_metadata"), "pinecone's Index API changed -- re-verify PINECONE_QUERY_METHODS"
    assert "search_records" in PINECONE_QUERY_METHODS
    assert "fetch_by_metadata" in PINECONE_QUERY_METHODS


def test_registered_pinecone_query_sinks_a_call_event_and_returns_the_real_result():
    from pinecone import Index

    original_query = Index.query
    Index.query = lambda self, **kwargs: {"matches": ["stand-in result"]}
    captured = []
    try:
        register_pinecone_callbacks(sink=captured.append)

        index = Index.__new__(Index)
        index.name = "my-index"
        with track_run("run-1"), track_step(4), track_node("retriever_tool"):
            result = index.query(vector=[0.1, 0.2], top_k=3)

        assert result == {"matches": ["stand-in result"]}
        assert len(captured) == 1
        event = captured[0]
        assert event.run_id == "run-1"
        assert event.step == 4
        assert event.caller == "retriever_tool"
        assert event.callee == "my-index"
        assert event.callee_type == NodeType.VECTOR_STORE
        assert event.error is False
    finally:
        Index.query = original_query


def test_registered_pinecone_query_skips_the_sink_when_run_context_is_not_set():
    from pinecone import Index

    original_query = Index.query
    Index.query = lambda self, **kwargs: {"matches": []}
    captured = []
    try:
        register_pinecone_callbacks(sink=captured.append)

        index = Index.__new__(Index)
        index.name = "my-index"
        result = index.query(vector=[0.1], top_k=1)  # no track_run/track_step block

        assert result == {"matches": []}
        assert captured == []
    finally:
        Index.query = original_query


def test_registered_pinecone_query_still_sinks_and_reraises_on_error():
    from pinecone import Index

    original_query = Index.query

    def _failing(self, **kwargs):
        raise RuntimeError("connection lost")

    Index.query = _failing
    captured = []
    try:
        register_pinecone_callbacks(sink=captured.append)

        index = Index.__new__(Index)
        index.name = "my-index"
        with track_run("run-1"), track_step(1), track_node("retriever_tool"), pytest.raises(RuntimeError):
            index.query(vector=[0.1], top_k=1)

        assert len(captured) == 1
        assert captured[0].error is True
    finally:
        Index.query = original_query


def test_register_weaviate_callbacks_patches_every_query_method():
    from weaviate.collections.collection.sync import _QueryCollection

    originals = {name: getattr(_QueryCollection, name) for name in WEAVIATE_QUERY_METHODS}
    try:
        register_weaviate_callbacks(sink=lambda event: None)
        for name in WEAVIATE_QUERY_METHODS:
            assert getattr(_QueryCollection, name) is not originals[name]
    finally:
        for name, original in originals.items():
            setattr(_QueryCollection, name, original)


def test_registered_weaviate_near_vector_sinks_a_call_event():
    from weaviate.collections.collection.sync import _QueryCollection

    original = _QueryCollection.near_vector
    _QueryCollection.near_vector = lambda self, *a, **kw: "stand-in result"
    captured = []
    try:
        register_weaviate_callbacks(sink=captured.append)

        query = _QueryCollection.__new__(_QueryCollection)
        query.name = "my-collection"
        with track_run("run-2"), track_step(7), track_node("retriever_tool"):
            result = query.near_vector([0.1, 0.2])

        assert result == "stand-in result"
        assert len(captured) == 1
        event = captured[0]
        assert event.run_id == "run-2"
        assert event.step == 7
        assert event.callee == "my-collection"
        assert event.error is False
    finally:
        _QueryCollection.near_vector = original
