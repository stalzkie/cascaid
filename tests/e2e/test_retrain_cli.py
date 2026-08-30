"""E2E seam: the real `cascaid retrain` CLI entry point, against a real Graph
Store on disk and a real sqlite DB holding IncidentLabel rows (PRD 4.3
fine-tuning on real customer data -- see docs/Real_Data_Retraining_Plan.md)."""

import sys
from datetime import datetime, timedelta, timezone

import pytest

import cascaid.retrain as retrain_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db, record_incident

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


def _seed(store_dir, database_url, run_ids):
    init_db(get_engine(database_url))
    session_factory = make_session_factory(database_url)
    with session_factory() as session:
        for i, run_id in enumerate(run_ids):
            base_time = T0 + timedelta(hours=i)
            events = [_event(run_id, 0, base_time), _event(run_id, 1, base_time + timedelta(minutes=10))]
            for snap in build_snapshots(NODES, EDGES, events):
                save_snapshot(to_pyg_data(snap), store_dir)
            record_incident(
                session,
                run_id=run_id,
                node_name="primary_model",
                incident_type="quality_degradation",
                occurred_at=base_time,
                source="manual",
            )


@pytest.mark.e2e
def test_retrain_cli_swaps_the_model_when_the_pr_auc_floor_is_cleared(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    _seed(store_dir, database_url, ["run-a", "run-b", "run-c", "run-d"])
    out_path = tmp_path / "model.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrain",
            "--database-url",
            database_url,
            "--store",
            str(store_dir),
            "--out",
            str(out_path),
            "--epochs",
            "2",
            "--min-pr-auc",
            "0.0",
        ],
    )

    retrain_cli.main()

    assert out_path.exists()
    assert "Swapped in the newly retrained model" in capsys.readouterr().out


@pytest.mark.e2e
def test_retrain_cli_does_not_swap_when_the_pr_auc_floor_is_not_cleared(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    _seed(store_dir, database_url, ["run-a", "run-b", "run-c", "run-d"])
    out_path = tmp_path / "model.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrain",
            "--database-url",
            database_url,
            "--store",
            str(store_dir),
            "--out",
            str(out_path),
            "--epochs",
            "2",
            "--min-pr-auc",
            "2.0",
        ],
    )

    retrain_cli.main()

    assert not out_path.exists()
    assert "keeping the existing model" in capsys.readouterr().out
