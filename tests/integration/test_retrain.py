"""Integration seam: cascaid.retrain.build_real_dataset/retrain against a real
Graph Store on disk -- the real-data counterpart to cascaid.train's synthetic
pipeline (see docs/Real_Data_Retraining_Plan.md)."""

from datetime import datetime, timedelta, timezone

import pytest

from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.retrain import build_real_dataset, retrain

NODES = {"agent": NodeType.AGENT, "primary_model": NodeType.MODEL_ENDPOINT}
EDGES = [("agent", "primary_model")]
T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)


def _event(run_id: str, step: int, occurred_at: datetime) -> CallEvent:
    return CallEvent(
        run_id=run_id,
        scenario="production",
        step=step,
        caller="agent",
        callee="primary_model",
        caller_type=NodeType.AGENT,
        callee_type=NodeType.MODEL_ENDPOINT,
        latency_ms=100.0,
        error=False,
        retried=False,
        token_cost=0.01,
        occurred_at=occurred_at,
    )


def _seed_run(store_dir, run_id: str, base_time: datetime):
    events = [_event(run_id, 0, base_time), _event(run_id, 1, base_time + timedelta(minutes=10))]
    snapshots = build_snapshots(NODES, EDGES, events)
    for snap in snapshots:
        save_snapshot(to_pyg_data(snap), store_dir)


@pytest.mark.integration
def test_build_real_dataset_labels_snapshots_from_incidents_within_window(tmp_path):
    store_dir = tmp_path / "store"
    _seed_run(store_dir, "run-1", T0)

    data_list = build_real_dataset(store_dir, ["run-1"], {"run-1": [("primary_model", T0)]})

    assert len(data_list) == 2
    step0 = next(d for d in data_list if d.step == 0)
    step1 = next(d for d in data_list if d.step == 1)
    assert step0.y[step0.node_order.index("primary_model")].item() == 1.0
    assert step1.y[step1.node_order.index("primary_model")].item() == 0.0
    assert bool(step0.usable.all())


@pytest.mark.integration
def test_build_real_dataset_returns_empty_for_a_run_with_no_snapshots(tmp_path):
    store_dir = tmp_path / "store"
    data_list = build_real_dataset(store_dir, ["missing-run"], {})
    assert data_list == []


@pytest.mark.integration
def test_retrain_trains_and_reports_metrics_on_real_data(tmp_path):
    store_dir = tmp_path / "store"
    # Every run gets one positive (incident at step 0) and one negative (step 1,
    # far outside the window) snapshot, so PR-AUC is well-defined regardless of
    # which run(s) split_run_ids happens to put in the held-out val set.
    for i, run_id in enumerate(["run-a", "run-b", "run-c", "run-d"]):
        _seed_run(store_dir, run_id, T0 + timedelta(hours=i))

    incidents_by_run = {
        run_id: [("primary_model", T0 + timedelta(hours=i))]
        for i, run_id in enumerate(["run-a", "run-b", "run-c", "run-d"])
    }

    model, metrics = retrain(store_dir, incidents_by_run, epochs=2, seed=0)

    assert model is not None
    assert metrics["num_train_snapshots"] + metrics["num_val_snapshots"] == 8
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["brier"] >= 0.0
