"""Integration seam: real FastAPI app wired to a real graph store + storage
database, exercised through TestClient (PRD 5.2 Dashboard API)."""

from datetime import datetime, timezone

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from torch_geometric.data import Data

from cascaid.dashboard.api import create_app
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.storage.repository import init_db, record_incident, record_scores


def _snapshot(run_id: str, step: int) -> Data:
    data = Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["agent", "store"]
    data.node_types = ["agent", "vector_store"]
    data.edges = [("agent", "store")]
    return data


def _app(tmp_path):
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    return create_app(store_dir=tmp_path, session_factory=session_factory), session_factory


@pytest.mark.integration
def test_runs_endpoint_returns_known_run_ids(tmp_path):
    save_snapshot(_snapshot("run-b", step=0), tmp_path)
    save_snapshot(_snapshot("run-a", step=0), tmp_path)
    app, _ = _app(tmp_path)
    client = TestClient(app)

    response = client.get("/runs")

    assert response.status_code == 200
    assert response.json() == {"run_ids": ["run-a", "run-b"]}


@pytest.mark.integration
def test_pipeline_endpoint_returns_nodes_and_edges(tmp_path):
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    app, session_factory = _app(tmp_path)
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.1, "store": 0.8})
    client = TestClient(app)

    response = client.get("/pipeline/run-1")

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == [
        {"name": "agent", "type": "agent", "risk_score": 0.1},
        {"name": "store", "type": "vector_store", "risk_score": 0.8},
    ]
    assert body["edges"] == [["agent", "store"]]


@pytest.mark.integration
def test_pipeline_endpoint_404_for_unknown_run(tmp_path):
    app, _ = _app(tmp_path)
    client = TestClient(app)

    response = client.get("/pipeline/no-such-run")

    assert response.status_code == 404


@pytest.mark.integration
def test_track_record_endpoint_returns_history_and_incidents(tmp_path):
    app, session_factory = _app(tmp_path)
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.3})
        record_incident(
            session,
            run_id="run-1",
            node_name="agent",
            incident_type="degradation",
            occurred_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
            source="manual",
        )
    client = TestClient(app)

    response = client.get("/track-record/run-1")

    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 1
    assert body["history"][0]["step"] == 0
    assert body["history"][0]["risk_score"] == 0.3
    assert "predicted_at" in body["history"][0]
    assert len(body["incidents"]) == 1
