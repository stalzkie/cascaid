"""Integration seam: GET /risk/{run_id}/explain/{node_name} against a real
FastAPI app + real CascadeGNN + real graph store, with a real local HTTP
server standing in for the configured LLM endpoint (PRD 7 risk explanations,
opt-in per docs/Client_Readiness_and_YC_Grade_Assessment.md section 4)."""

import pytest
import torch
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from torch_geometric.data import Data

from cascaid.auth.passwords import hash_password
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.api import create_app
from cascaid.storage.repository import init_db, set_config
from cascaid.storage.secrets import set_secret_config

IN_DIM = 6
EDGE_DIM = 4


@pytest.fixture(autouse=True)
def _config_encryption_key(monkeypatch):
    # ADR 0005: llm_api_key is encrypted at rest -- every test in this file that
    # configures it needs a key present, even the ones that don't (harmless).
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())


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
    data.edges = [("a", "b"), ("b", "c")]
    return data


def _session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    return sessionmaker(bind=engine)


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
def test_explain_endpoint_returns_llm_text_when_enabled(tmp_path, httpserver):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    session_factory = _session_factory()
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"content": "b is at risk because of c."}}]}
    )
    with session_factory() as session:
        set_config(session, "llm_explanations_enabled", "true")
        set_config(session, "llm_base_url", httpserver.url_for("/v1"))
        set_secret_config(session, "llm_api_key", "sk-test")
        set_config(session, "llm_model", "gpt-4o-mini")
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)

    response = client.get("/risk/run-1/explain/b", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["node_name"] == "b"
    assert body["explanation"] == "b is at risk because of c."


@pytest.mark.integration
def test_explain_endpoint_404s_when_not_enabled(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    session_factory = _session_factory()
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)

    response = client.get("/risk/run-1/explain/b", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404


@pytest.mark.integration
def test_explain_endpoint_404s_for_unknown_node(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    session_factory = _session_factory()
    with session_factory() as session:
        set_config(session, "llm_explanations_enabled", "true")
        set_config(session, "llm_base_url", "http://unused")
        set_secret_config(session, "llm_api_key", "sk-test")
        set_config(session, "llm_model", "gpt-4o-mini")
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)

    response = client.get("/risk/run-1/explain/not-a-real-node", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 404


@pytest.mark.integration
def test_explain_endpoint_503s_when_the_llm_endpoint_is_unreachable(tmp_path):
    torch.manual_seed(0)
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    session_factory = _session_factory()
    with session_factory() as session:
        set_config(session, "llm_explanations_enabled", "true")
        set_config(session, "llm_base_url", "http://127.0.0.1:1/v1")
        set_secret_config(session, "llm_api_key", "sk-test")
        set_config(session, "llm_model", "gpt-4o-mini")
    app = create_app(model=model, store_dir=tmp_path, session_factory=session_factory)
    client = TestClient(app)
    token = _login(client, session_factory)

    response = client.get("/risk/run-1/explain/b", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 503
