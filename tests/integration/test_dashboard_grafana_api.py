"""Integration seam: the /grafana/* routes on the real dashboard FastAPI app,
implementing the SimPod-json-datasource protocol (PRD 4.7)."""

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
from cascaid.storage.repository import init_db, record_scores, set_config


def _snapshot(run_id: str, step: int) -> Data:
    data = Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["agent", "store"]
    data.node_types = ["agent", "vector_store"]
    data.edges = [("agent", "store")]
    return data


def _app_with_admin(tmp_path, username="admin", password="hunter2"):
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        set_config(session, "auth_username", username)
        set_config(session, "auth_password_hash", hash_password(password))
    return create_app(store_dir=tmp_path, session_factory=session_factory), session_factory


def _login(client, username="admin", password="hunter2") -> str:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["token"]


@pytest.mark.integration
def test_grafana_root_401s_without_a_token(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)

    response = client.get("/grafana/")

    assert response.status_code == 401


@pytest.mark.integration
def test_grafana_root_200s_with_a_valid_token(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    response = client.get("/grafana/", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200


@pytest.mark.integration
def test_grafana_search_returns_known_targets(tmp_path):
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    response = client.post("/grafana/search", json={"target": ""}, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == ["run-1/agent", "run-1/store"]


@pytest.mark.integration
def test_grafana_query_returns_datapoints_for_a_known_target(tmp_path):
    app, session_factory = _app_with_admin(tmp_path)
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.4})
    client = TestClient(app)
    token = _login(client)

    response = client.post(
        "/grafana/query",
        json={"targets": [{"target": "run-1/agent"}], "range": {"from": "now-1h", "to": "now"}},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body[0]["target"] == "run-1/agent"
    assert [value for value, _ in body[0]["datapoints"]] == [0.4]


@pytest.mark.integration
def test_grafana_query_400s_for_a_malformed_target(tmp_path):
    app, _ = _app_with_admin(tmp_path)
    client = TestClient(app)
    token = _login(client)

    response = client.post(
        "/grafana/query",
        json={"targets": [{"target": "agent"}]},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
