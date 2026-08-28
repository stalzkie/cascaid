"""Integration seam: real FastAPI app wired to a real CascadeGNN and a real
graph store on disk, exercised through TestClient (PRD 5.2 Model Serving)."""

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from torch_geometric.data import Data

from cascaid.ingestion.graph_store import save_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.api import create_app
from cascaid.storage.repository import get_score_history, init_db

IN_DIM = 6
EDGE_DIM = 4


def _snapshot(run_id: str, step: int) -> Data:
    data = Data(
        x=torch.rand(3, IN_DIM),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.rand(2, EDGE_DIM),
    )
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["a", "b", "c"]
    return data


@pytest.mark.integration
def test_health_endpoint_returns_ok(tmp_path):
    torch.manual_seed(0)
    app = create_app(model=None, store_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_risk_endpoint_returns_scores_for_latest_snapshot(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    save_snapshot(_snapshot("run-1", step=1), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    app = create_app(model=model, store_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/risk/run-1")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["step"] == 1
    assert set(body["scores"].keys()) == {"a", "b", "c"}
    assert all(0.0 <= v <= 1.0 for v in body["scores"].values())


@pytest.mark.integration
def test_risk_endpoint_404_for_unknown_run(tmp_path):
    app = create_app(model=None, store_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/risk/no-such-run")

    assert response.status_code == 404


@pytest.mark.integration
def test_risk_endpoint_persists_scores_when_session_factory_given(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)

    response = client.get("/risk/run-1")
    assert response.status_code == 200

    with session_factory() as session:
        history = get_score_history(session, run_id="run-1")
    assert {row.node_name for row in history} == {"a", "b", "c"}
    assert {row.risk_score for row in history} == set(response.json()["scores"].values())
