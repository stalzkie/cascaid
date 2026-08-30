from datetime import datetime, timedelta, timezone

from cascaid.ingestion.schema import CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data

NODES = {"agent": NodeType.AGENT, "store": NodeType.VECTOR_STORE}
EDGES = [("agent", "store")]


def _event(step: int, occurred_at: datetime | None = None) -> CallEvent:
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
        occurred_at=occurred_at,
    )


def test_to_pyg_data_carries_node_types_alongside_node_order():
    snapshots = build_snapshots(NODES, EDGES, [_event(0)])
    data = to_pyg_data(snapshots[0])

    assert data.node_order == ["agent", "store"]
    assert data.node_types == ["agent", "vector_store"]


def test_to_pyg_data_carries_directed_edges_by_node_name():
    snapshots = build_snapshots(NODES, EDGES, [_event(0)])
    data = to_pyg_data(snapshots[0])

    assert data.edges == [("agent", "store")]


def test_build_snapshots_derives_step_wall_clock_bounds_from_occurred_at():
    t0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1)
    snapshots = build_snapshots(NODES, EDGES, [_event(0, occurred_at=t0), _event(0, occurred_at=t1)])

    assert snapshots[0].step_start == t0
    assert snapshots[0].step_end == t1


def test_build_snapshots_leaves_step_bounds_none_without_occurred_at():
    snapshots = build_snapshots(NODES, EDGES, [_event(0)])

    assert snapshots[0].step_start is None
    assert snapshots[0].step_end is None


def test_to_pyg_data_carries_step_wall_clock_bounds():
    t0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    snapshots = build_snapshots(NODES, EDGES, [_event(0, occurred_at=t0)])
    data = to_pyg_data(snapshots[0])

    assert data.step_start == t0
    assert data.step_end == t0
