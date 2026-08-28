"""E2E seam: the real `cascaid.dashboard.serve` CLI, reading back a snapshot +
scores produced by the real demo pipeline and storage repository."""

import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

import cascaid.dashboard.serve as dashboard_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db, record_scores
from cascaid_demo.fault_injection import make_scenario
from cascaid_demo.mock_llm_gateway import ModelGateway
from cascaid_demo.mock_vector_db import VectorStore
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES, build_pipeline
from cascaid_demo.recorder import Recorder

TOTAL_STEPS = 10


@pytest.mark.e2e
def test_dashboard_cli_serves_pipeline_and_track_record(tmp_path, monkeypatch):
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    rng = np.random.default_rng(5)
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
                "run_id": "dashboard-e2e-run",
            }
        }
        state = graph.invoke(state, config=config)
    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    with make_session_factory(database_url)() as session:
        for snap in snapshots:
            data = to_pyg_data(snap)
            save_snapshot(data, store_dir)
            record_scores(
                session, run_id="dashboard-e2e-run", step=snap.step, scores=dict.fromkeys(data.node_order, 0.1)
            )

    monkeypatch.setattr(sys, "argv", ["dashboard", "--database-url", database_url, "--store", str(store_dir)])
    app = dashboard_cli.build_app_from_argv()
    client = TestClient(app)

    pipeline_response = client.get("/pipeline/dashboard-e2e-run")
    track_record_response = client.get("/track-record/dashboard-e2e-run")

    assert pipeline_response.status_code == 200
    assert {n["name"] for n in pipeline_response.json()["nodes"]} == set(STATIC_NODES.keys())
    assert track_record_response.status_code == 200
    assert len(track_record_response.json()["history"]) == TOTAL_STEPS * len(STATIC_NODES)
