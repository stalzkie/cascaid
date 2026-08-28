"""Integration seam: real build_snapshots() -> to_pyg_data() output persisted through
the Graph Store and read back (PRD 5.2 Graph Store feeding Model Serving)."""

import pytest
import torch

from cascaid.ingestion.graph_store import latest_snapshot, save_snapshot
from cascaid.ingestion.schema import CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data

NODES = {"agent": NodeType.AGENT, "model": NodeType.MODEL_ENDPOINT}
EDGES = [("agent", "model")]


def _event(step: int) -> CallEvent:
    return CallEvent(
        run_id="persist-run",
        scenario="baseline",
        step=step,
        caller="agent",
        callee="model",
        caller_type=NodeType.AGENT,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=100.0 + step,
        error=False,
        retried=False,
        token_cost=0.01,
    )


@pytest.mark.integration
def test_latest_persisted_snapshot_matches_final_build_snapshots_step(tmp_path):
    events = [_event(step) for step in range(4)]
    snapshots = build_snapshots(NODES, EDGES, events)

    for snap in snapshots:
        data = to_pyg_data(snap)
        save_snapshot(data, tmp_path)

    loaded = latest_snapshot(tmp_path, "persist-run")
    expected = to_pyg_data(snapshots[-1])

    assert loaded.step == expected.step == 3
    assert torch.equal(loaded.x, expected.x)
    assert torch.equal(loaded.edge_attr, expected.edge_attr)
