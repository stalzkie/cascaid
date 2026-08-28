import time

import pytest

from cascaid.ingestion.schema import NodeType
from cascaid.ingestion.vector_query_adapter import observe_vector_query


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


def test_observe_vector_query_records_error_and_reraises():
    tracker_ref = {}

    with pytest.raises(RuntimeError, match="connection lost"):
        with observe_vector_query("retriever_tool", "vector_store", run_id="run-1", step=2) as tracker:
            tracker_ref["tracker"] = tracker
            raise RuntimeError("connection lost")

    event = tracker_ref["tracker"].event
    assert event.error is True
