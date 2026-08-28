"""E2E seam: run_scenarios -> train -> serve, wired end-to-end through the real
CLI entry points -- a trained model and a persisted snapshot are read back through
the actual FastAPI app the `cascaid.serve` CLI builds from argv."""

import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

import cascaid.serve as serve_cli
import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_score_history, init_db
from cascaid_demo.fault_injection import make_scenario
from cascaid_demo.mock_llm_gateway import ModelGateway
from cascaid_demo.mock_vector_db import VectorStore
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES, build_pipeline
from cascaid_demo.recorder import Recorder

TOTAL_STEPS = 10


@pytest.mark.e2e
def test_serve_cli_serves_risk_for_a_persisted_snapshot(tmp_path, monkeypatch):
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"
    store_dir = tmp_path / "graph_store"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scenarios", "--runs-per-scenario", "3", "--steps", "20", "--out", str(data_dir), "--seed", "0"],
    )
    run_scenarios_cli.main()

    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data_dir), "--epochs", "3", "--out", str(model_path)])
    train_cli.main()
    assert model_path.exists()

    rng = np.random.default_rng(3)
    scenario = make_scenario("baseline", TOTAL_STEPS, rng)
    recorder = Recorder()
    graph = build_pipeline()
    vector_store = VectorStore()
    gateway = ModelGateway()
    state = {"query": "", "retrieved_context": "", "research_notes": "", "answer": ""}
    for step in range(TOTAL_STEPS):
        config = {
            "configurable": {
                "recorder": recorder,
                "scenario": scenario,
                "step": step,
                "rng": rng,
                "vector_store": vector_store,
                "gateway": gateway,
                "run_id": "serve-e2e-run",
            }
        }
        state = graph.invoke(state, config=config)
    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    for snap in snapshots:
        save_snapshot(to_pyg_data(snap), store_dir)

    db_path = tmp_path / "cascaid.db"
    database_url = f"sqlite:///{db_path}"
    init_db(get_engine(database_url))
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve", "--model", str(model_path), "--store", str(store_dir), "--database-url", database_url],
    )
    app = serve_cli.build_app_from_argv()
    client = TestClient(app)

    response = client.get("/risk/serve-e2e-run")

    assert response.status_code == 200
    body = response.json()
    assert body["step"] == TOTAL_STEPS - 1
    assert set(body["scores"].keys()) == set(STATIC_NODES.keys())

    with make_session_factory(database_url)() as session:
        history = get_score_history(session, run_id="serve-e2e-run")
    assert {row.node_name for row in history} == set(STATIC_NODES.keys())
