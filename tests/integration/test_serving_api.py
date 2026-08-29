"""Integration seam: real FastAPI app wired to a real CascadeGNN and a real
graph store on disk, exercised through TestClient (PRD 5.2 Model Serving)."""

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from torch_geometric.data import Data

from cascaid.auth.passwords import hash_password
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.api import create_app
from cascaid.storage.repository import get_alert_history, get_score_history, init_db, set_config

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
    data.node_types = ["agent", "tool", "vector_store"]
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


def _login(client, session_factory, username="admin", password="hunter2") -> str:
    with session_factory() as session:
        set_config(session, "auth_username", username)
        set_config(session, "auth_password_hash", hash_password(password))
    from cascaid.dashboard.api import create_app as create_dashboard_app

    dashboard_app = create_dashboard_app(store_dir="unused", session_factory=session_factory)
    response = TestClient(dashboard_app).post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["token"]


@pytest.mark.integration
def test_risk_endpoint_401s_without_a_token_when_persistence_is_configured(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)

    response = client.get("/risk/run-1")

    assert response.status_code == 401


@pytest.mark.integration
def test_risk_endpoint_is_unauthenticated_when_no_database_is_configured(tmp_path):
    # Deliberate: with no --database-url there is nowhere to store a validatable
    # session, so cascaid serve without persistence stays open, same as today.
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    app = create_app(model=model, store_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/risk/run-1")

    assert response.status_code == 200


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
    token = _login(client, session_factory)

    response = client.get("/risk/run-1", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200

    with session_factory() as session:
        history = get_score_history(session, run_id="run-1")
    assert {row.node_name for row in history} == {"a", "b", "c"}
    assert {row.risk_score for row in history} == set(response.json()["scores"].values())


@pytest.mark.integration
def test_risk_endpoint_fires_alert_when_enabled_and_threshold_crossed(tmp_path, httpserver):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        set_config(session, "alerting_enabled", "true")
        set_config(session, "alert_threshold", "0.0")
        set_config(session, "alert_webhook_url", httpserver.url_for("/hook"))
    httpserver.expect_request("/hook", method="POST").respond_with_json({"ok": True})

    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)
    response = client.get("/risk/run-1", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200  # all 3 nodes score >= 0.0 threshold, all fire

    with session_factory() as session:
        alerts = get_alert_history(session, run_id="run-1")
    assert len(alerts) == 3
    assert {a.node_name for a in alerts} == {"a", "b", "c"}


@pytest.mark.integration
def test_risk_endpoint_does_not_alert_when_alerting_disabled(tmp_path, httpserver):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        set_config(session, "alert_threshold", "0.0")
        set_config(session, "alert_webhook_url", httpserver.url_for("/hook"))

    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)
    response = client.get("/risk/run-1", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200

    with session_factory() as session:
        alerts = get_alert_history(session, run_id="run-1")
    assert alerts == []
