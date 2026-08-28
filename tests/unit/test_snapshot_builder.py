from cascaid.ingestion.schema import CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data

NODES = {"agent": NodeType.AGENT, "store": NodeType.VECTOR_STORE}
EDGES = [("agent", "store")]


def _event(step: int) -> CallEvent:
    return CallEvent(
        run_id="run-1",
        scenario="baseline",
        step=step,
        caller="agent",
        callee="store",
        caller_type=NodeType.AGENT,
        callee_type=NodeType.VECTOR_STORE,
        latency_ms=50.0,
        error=False,
        retried=False,
        token_cost=0.0,
    )


def test_to_pyg_data_carries_node_types_alongside_node_order():
    snapshots = build_snapshots(NODES, EDGES, [_event(0)])
    data = to_pyg_data(snapshots[0])

    assert data.node_order == ["agent", "store"]
    assert data.node_types == ["agent", "vector_store"]
