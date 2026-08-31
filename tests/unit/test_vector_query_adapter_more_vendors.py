"""Chroma/Qdrant/Milvus/LanceDB coverage for vector_query_adapter.py -- kept in a
separate file from test_vector_query_adapter.py's Pinecone/Weaviate tests purely for
file size; same "real classes/method names, a stand-in only at the network-call layer"
philosophy (see that file's own comment on why)."""

import asyncio
import types

import pytest

import cascaid.ingestion.vector_query_adapter as vector_query_adapter
from cascaid.ingestion.runtime_context import track_node, track_run, track_step
from cascaid.ingestion.schema import NodeType
from cascaid.ingestion.vector_query_adapter import (
    CHROMA_QUERY_METHODS,
    MILVUS_ASYNC_QUERY_METHODS,
    MILVUS_QUERY_METHODS,
    QDRANT_QUERY_METHODS,
    register_chroma_callbacks,
    register_lancedb_callbacks,
    register_milvus_callbacks,
    register_qdrant_callbacks,
)


@pytest.fixture(autouse=True)
def _reset_patch_state():
    for attr in (
        "_patched_chroma_sink",
        "_patched_chroma_async_sink",
        "_patched_qdrant_sink",
        "_patched_qdrant_async_sink",
        "_patched_milvus_sink",
        "_patched_milvus_async_sink",
        "_patched_lancedb_sinks",
    ):
        setattr(vector_query_adapter, attr, None)
    yield
    for attr in (
        "_patched_chroma_sink",
        "_patched_chroma_async_sink",
        "_patched_qdrant_sink",
        "_patched_qdrant_async_sink",
        "_patched_milvus_sink",
        "_patched_milvus_async_sink",
        "_patched_lancedb_sinks",
    ):
        setattr(vector_query_adapter, attr, None)


# --- Chroma ---


def test_register_chroma_callbacks_patches_every_query_method_on_both_clients():
    from chromadb.api.models.AsyncCollection import AsyncCollection
    from chromadb.api.models.Collection import Collection

    sync_originals = {name: getattr(Collection, name) for name in CHROMA_QUERY_METHODS}
    async_originals = {name: getattr(AsyncCollection, name) for name in CHROMA_QUERY_METHODS}
    try:
        register_chroma_callbacks(sink=lambda event: None)
        for name in CHROMA_QUERY_METHODS:
            assert getattr(Collection, name) is not sync_originals[name]
            assert getattr(AsyncCollection, name) is not async_originals[name]
    finally:
        for name, original in sync_originals.items():
            setattr(Collection, name, original)
        for name, original in async_originals.items():
            setattr(AsyncCollection, name, original)


def test_registered_chroma_query_sinks_a_call_event_and_returns_the_real_result():
    from chromadb.api.models.Collection import Collection

    original_query = Collection.query
    Collection.query = lambda self, **kwargs: {"ids": [["a", "b"]]}
    captured = []
    try:
        register_chroma_callbacks(sink=captured.append)

        collection = Collection.__new__(Collection)
        collection._model = types.SimpleNamespace(name="my-collection")  # Collection.name is a read-only property
        with track_run("run-1"), track_step(1), track_node("retriever_tool"):
            result = collection.query(query_texts=["hi"], n_results=2)

        assert result == {"ids": [["a", "b"]]}
        assert len(captured) == 1
        event = captured[0]
        assert event.run_id == "run-1"
        assert event.step == 1
        assert event.caller == "retriever_tool"
        assert event.callee == "my-collection"
        assert event.callee_type == NodeType.VECTOR_STORE
        assert event.error is False
    finally:
        Collection.query = original_query


def test_registered_chroma_async_query_sinks_a_call_event():
    from chromadb.api.models.AsyncCollection import AsyncCollection

    original_query = AsyncCollection.query

    async def fake_query(self, **kwargs):
        return {"ids": [["a"]]}

    AsyncCollection.query = fake_query
    captured = []
    try:
        register_chroma_callbacks(sink=captured.append)

        collection = AsyncCollection.__new__(AsyncCollection)
        collection._model = types.SimpleNamespace(name="my-collection")
        with track_run("run-1"), track_step(1), track_node("retriever_tool"):
            result = asyncio.run(collection.query(query_texts=["hi"]))

        assert result == {"ids": [["a"]]}
        assert len(captured) == 1
        assert captured[0].callee == "my-collection"
    finally:
        AsyncCollection.query = original_query


# --- Qdrant ---


def test_register_qdrant_callbacks_patches_every_query_method_on_both_clients():
    from qdrant_client import AsyncQdrantClient, QdrantClient

    sync_originals = {name: getattr(QdrantClient, name) for name in QDRANT_QUERY_METHODS}
    async_originals = {name: getattr(AsyncQdrantClient, name) for name in QDRANT_QUERY_METHODS}
    try:
        register_qdrant_callbacks(sink=lambda event: None)
        for name in QDRANT_QUERY_METHODS:
            assert getattr(QdrantClient, name) is not sync_originals[name]
            assert getattr(AsyncQdrantClient, name) is not async_originals[name]
    finally:
        for name, original in sync_originals.items():
            setattr(QdrantClient, name, original)
        for name, original in async_originals.items():
            setattr(AsyncQdrantClient, name, original)


def test_registered_qdrant_query_points_sinks_a_call_event_with_callee_from_collection_name_kwarg():
    # QdrantClient isn't per-collection -- callee must come from the collection_name
    # kwarg, not a `self.name` attribute (QdrantClient has none).
    from qdrant_client import QdrantClient

    original = QdrantClient.query_points
    QdrantClient.query_points = lambda self, collection_name, **kwargs: {"points": []}
    captured = []
    try:
        register_qdrant_callbacks(sink=captured.append)

        client = QdrantClient.__new__(QdrantClient)
        with track_run("run-1"), track_step(2), track_node("retriever_tool"):
            result = client.query_points(collection_name="my-collection", query=[0.1, 0.2])

        assert result == {"points": []}
        assert len(captured) == 1
        event = captured[0]
        assert event.callee == "my-collection"
        assert event.error is False
    finally:
        QdrantClient.query_points = original


def test_registered_qdrant_query_points_sinks_and_reraises_on_error():
    from qdrant_client import QdrantClient

    original = QdrantClient.query_points

    def _failing(self, collection_name, **kwargs):
        raise RuntimeError("connection lost")

    QdrantClient.query_points = _failing
    captured = []
    try:
        register_qdrant_callbacks(sink=captured.append)

        client = QdrantClient.__new__(QdrantClient)
        with track_run("run-1"), track_step(1), track_node("retriever_tool"), pytest.raises(RuntimeError):
            client.query_points(collection_name="my-collection", query=[0.1])

        assert len(captured) == 1
        assert captured[0].error is True
        assert captured[0].callee == "my-collection"
    finally:
        QdrantClient.query_points = original


# --- Milvus ---


def test_register_milvus_callbacks_patches_sync_and_a_narrower_async_method_set():
    from pymilvus import AsyncMilvusClient, MilvusClient

    sync_originals = {name: getattr(MilvusClient, name) for name in MILVUS_QUERY_METHODS}
    async_originals = {name: getattr(AsyncMilvusClient, name) for name in MILVUS_ASYNC_QUERY_METHODS}
    try:
        register_milvus_callbacks(sink=lambda event: None)
        for name in MILVUS_QUERY_METHODS:
            assert getattr(MilvusClient, name) is not sync_originals[name]
        for name in MILVUS_ASYNC_QUERY_METHODS:
            assert getattr(AsyncMilvusClient, name) is not async_originals[name]
    finally:
        for name, original in sync_originals.items():
            setattr(MilvusClient, name, original)
        for name, original in async_originals.items():
            setattr(AsyncMilvusClient, name, original)


def test_registered_milvus_search_sinks_a_call_event_with_callee_from_collection_name_kwarg():
    from pymilvus import MilvusClient

    original = MilvusClient.search
    MilvusClient.search = lambda self, collection_name, **kwargs: [[]]
    captured = []
    try:
        register_milvus_callbacks(sink=captured.append)

        client = MilvusClient.__new__(MilvusClient)
        with track_run("run-1"), track_step(3), track_node("retriever_tool"):
            result = client.search(collection_name="my-collection", data=[[0.1, 0.2]])

        assert result == [[]]
        assert len(captured) == 1
        assert captured[0].callee == "my-collection"
        assert captured[0].error is False
    finally:
        MilvusClient.search = original


# --- LanceDB ---


def test_register_lancedb_callbacks_patches_to_arrow_on_every_concrete_query_builder():
    from lancedb.query import (
        LanceEmptyQueryBuilder,
        LanceFtsQueryBuilder,
        LanceHybridQueryBuilder,
        LanceVectorQueryBuilder,
    )

    builder_classes = [LanceVectorQueryBuilder, LanceFtsQueryBuilder, LanceHybridQueryBuilder, LanceEmptyQueryBuilder]
    originals = {cls: cls.__dict__["to_arrow"] for cls in builder_classes}
    try:
        register_lancedb_callbacks(sink=lambda event: None)
        for cls in builder_classes:
            assert cls.__dict__["to_arrow"] is not originals[cls]
    finally:
        for cls, original in originals.items():
            cls.to_arrow = original


def test_registered_lancedb_to_arrow_sinks_a_call_event_and_returns_the_real_result():
    # Proves the "patch to_arrow once, cover every terminal method" design: a customer
    # calling table.search(...).to_pandas() (which delegates to to_arrow internally)
    # produces exactly one CallEvent, not zero and not a stale/near-instant one from
    # search() itself (which never touches the store -- see register_lancedb_callbacks'
    # docstring).
    from lancedb.query import LanceVectorQueryBuilder

    original = LanceVectorQueryBuilder.__dict__["to_arrow"]
    LanceVectorQueryBuilder.to_arrow = lambda self, **kwargs: "stand-in arrow table"
    captured = []
    try:
        register_lancedb_callbacks(sink=captured.append)

        builder = LanceVectorQueryBuilder.__new__(LanceVectorQueryBuilder)
        with track_run("run-1"), track_step(5), track_node("retriever_tool"):
            result = builder.to_arrow()

        assert result == "stand-in arrow table"
        assert len(captured) == 1
        event = captured[0]
        assert event.caller == "retriever_tool"
        assert event.callee == "lancedb"  # documented fallback -- no accessible table name
        assert event.callee_type == NodeType.VECTOR_STORE
        assert event.error is False
    finally:
        LanceVectorQueryBuilder.to_arrow = original


def test_registered_lancedb_to_arrow_sinks_and_reraises_on_error():
    from lancedb.query import LanceFtsQueryBuilder

    original = LanceFtsQueryBuilder.__dict__["to_arrow"]

    def _failing(self, **kwargs):
        raise RuntimeError("connection lost")

    LanceFtsQueryBuilder.to_arrow = _failing
    captured = []
    try:
        register_lancedb_callbacks(sink=captured.append)

        builder = LanceFtsQueryBuilder.__new__(LanceFtsQueryBuilder)
        with track_run("run-1"), track_step(1), track_node("retriever_tool"), pytest.raises(RuntimeError):
            builder.to_arrow()

        assert len(captured) == 1
        assert captured[0].error is True
    finally:
        LanceFtsQueryBuilder.to_arrow = original
