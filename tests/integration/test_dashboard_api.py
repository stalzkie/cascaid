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

from cascaid.auth.passwords import hash_password
from cascaid.dashboard.api import create_app
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.storage.repository import init_db, record_incident, record_scores, set_config


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


def _app_with_admin(tmp_path, username="admin", password="hunter2"):
    app, session_factory = _app(tmp_path)
    with session_factory() as session:
        set_config(session, "auth_username", username)
        set_config(session, "auth_password_hash", hash_password(password))
    return app, session_factory


def _login(client, username="admin", password="hunter2") -> str:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["token"]


@pytest.mark.integration
def test_runs_endpoint_returns_known_run_ids(tmp_path):
    save_snapshot(_snapshot("run-b", step=0), tmp_path)
    save_snapshot(_snapshot("run-a", step=0), tmp_path)
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    response = client.get("/runs", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"run_ids": ["run-a", "run-b"]}


@pytest.mark.integration
def test_pipeline_endpoint_returns_nodes_and_edges(tmp_path):
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    app, session_factory = _app_with_admin(tmp_path)
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.1, "store": 0.8})
    client = TestClient(app)
    token = _login(client)

    response = client.get("/pipeline/run-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"] == [
        {"name": "agent", "type": "agent", "risk_score": 0.1},
        {"name": "store", "type": "vector_store", "risk_score": 0.8},
    ]
    assert body["edges"] == [["agent", "store"]]


@pytest.mark.integration
def test_pipeline_endpoint_404_for_unknown_run(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    response = client.get("/pipeline/no-such-run", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404


@pytest.mark.integration
def test_track_record_endpoint_returns_history_and_incidents(tmp_path):
    app, session_factory = _app_with_admin(tmp_path)
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
    token = _login(client)

    response = client.get("/track-record/run-1", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 1
    assert body["history"][0]["step"] == 0
    assert body["history"][0]["risk_score"] == 0.3
    assert "predicted_at" in body["history"][0]
    assert len(body["incidents"]) == 1


@pytest.mark.integration
def test_protected_endpoint_401s_without_a_token(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)

    response = client.get("/runs")

    assert response.status_code == 401


@pytest.mark.integration
def test_protected_endpoint_401s_with_an_invalid_token(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)

    response = client.get("/runs", headers={"Authorization": "Bearer not-a-real-token"})

    assert response.status_code == 401


@pytest.mark.integration
def test_login_rejects_the_wrong_password(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)

    response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})

    assert response.status_code == 401


@pytest.mark.integration
def test_logout_invalidates_the_token(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    logout_response = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_response.status_code == 200

    response = client.get("/runs", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
