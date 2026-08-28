"""Integration seam: real demo-run data (run_scenarios output) -> graph store +
score history, via cascaid_demo.seed_store.seed(). This is what makes `docker
compose up` show something in the dashboard on first run (PRD 4.2 local demo mode)
instead of an empty graph store no live pipeline has ever written to."""

import sys

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import cascaid_demo.run_scenarios as run_scenarios_cli
from cascaid.ingestion.graph_store import latest_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.storage.repository import get_score_history, init_db
from cascaid_demo.seed_store import seed

TOTAL_STEPS = 20


@pytest.mark.integration
def test_seed_persists_a_snapshot_and_scores_for_every_demo_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "runs"
    store_dir = tmp_path / "graph_store"
    model_path = tmp_path / "pretrained_base.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scenarios",
            "--runs-per-scenario",
            "2",
            "--steps",
            str(TOTAL_STEPS),
            "--out",
            str(data_dir),
            "--seed",
            "0",
        ],
    )
    run_scenarios_cli.main()

    torch.manual_seed(0)
    model = CascadeGNN(in_dim=8, edge_dim=4, hidden=8)
    torch.save(model.state_dict(), model_path)

    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)

    counts = seed(
        data_dir=data_dir, model_path=model_path, store_dir=store_dir, session_factory=session_factory, hidden=8
    )

    assert counts["runs"] > 0
    assert counts["snapshots"] == counts["runs"] * TOTAL_STEPS

    with session_factory() as session:
        seeded_run_id = next(iter(counts["run_ids"]))
        history = get_score_history(session, run_id=seeded_run_id)
    assert len(history) > 0

    snapshot = latest_snapshot(store_dir, seeded_run_id)
    assert snapshot is not None
    assert snapshot.step == TOTAL_STEPS - 1
