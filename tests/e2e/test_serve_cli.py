"""E2E seam: run_scenarios -> train -> serve, wired end-to-end through the real
CLI entry points -- a trained model and a persisted snapshot are read back through
the actual FastAPI app the `cascaid.serve` CLI builds from argv."""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

import cascaid.serve as serve_cli
import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import create_session, get_score_history
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
    # No init_db() call here on purpose -- the CLI must initialize its own schema
    # (docker compose up shouldn't need a separate migration step, PRD 4.4).
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve", "--model", str(model_path), "--store", str(store_dir), "--database-url", database_url],
    )
    app = serve_cli.build_app_from_argv()
    client = TestClient(app)
    with make_session_factory(database_url)() as session:
        create_session(session, token="test-token", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    response = client.get("/risk/serve-e2e-run", headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 200
    body = response.json()
    assert body["step"] == TOTAL_STEPS - 1
    assert set(body["scores"].keys()) == set(STATIC_NODES.keys())

    with make_session_factory(database_url)() as session:
        history = get_score_history(session, run_id="serve-e2e-run")
    assert {row.node_name for row in history} == set(STATIC_NODES.keys())


@pytest.mark.e2e
def test_serve_cli_loads_a_model_trained_with_non_default_hyperparameters_with_zero_flags(tmp_path, monkeypatch):
    # Regression test for the real gap this closes: before the sidecar .config.json
    # (models/model_config.py), a model trained with non-default hidden/layers/conv --
    # exactly what scripts/gnn_experiment.py's own accuracy sweeps produce, see
    # docs/GNN_Accuracy_Improvement_Log.md -- would fail to load under `cascaid serve`'s
    # defaults with a raw PyTorch "size mismatch" error, not a clear one. `cascaid serve`
    # with zero --hidden/--layers/--conv flags must now pick up the trained
    # configuration automatically.
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"
    store_dir = tmp_path / "graph_store"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scenarios", "--runs-per-scenario", "3", "--steps", "20", "--out", str(data_dir), "--seed", "0"],
    )
    run_scenarios_cli.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(data_dir),
            "--epochs",
            "3",
            "--out",
            str(model_path),
            "--hidden",
            "16",
            "--layers",
            "3",
            "--conv",
            "gat",
        ],
    )
    train_cli.main()
    assert model_path.with_suffix(".config.json").exists()

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
                "run_id": "non-default-hparams-run",
            }
        }
        state = graph.invoke(state, config=config)
    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    for snap in snapshots:
        save_snapshot(to_pyg_data(snap), store_dir)

    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve", "--model", str(model_path), "--store", str(store_dir), "--database-url", database_url],
    )
    app = serve_cli.build_app_from_argv()
    client = TestClient(app)
    with make_session_factory(database_url)() as session:
        create_session(session, token="test-token", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    response = client.get("/risk/non-default-hparams-run", headers={"Authorization": "Bearer test-token"})

    assert response.status_code == 200
    assert set(response.json()["scores"].keys()) == set(STATIC_NODES.keys())


@pytest.mark.e2e
def test_serve_cli_still_requires_auth_when_database_url_is_omitted(tmp_path, monkeypatch):
    # Security regression test: `cascaid serve` with no --database-url is exactly
    # the invocation shown in serve.py's own module docstring, and used to leave
    # /risk/{run_id} completely unauthenticated (no place to store a session token).
    # It must now provision its own store and keep auth enforced by default.
    model_path = tmp_path / "models" / "pretrained_base.pt"
    store_dir = tmp_path / "graph_store"
    data_dir = tmp_path / "runs"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scenarios", "--runs-per-scenario", "3", "--steps", "20", "--out", str(data_dir), "--seed", "0"],
    )
    run_scenarios_cli.main()
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data_dir), "--epochs", "3", "--out", str(model_path)])
    train_cli.main()

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
                "run_id": "no-database-url-run",
            }
        }
        state = graph.invoke(state, config=config)
    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    for snap in snapshots:
        save_snapshot(to_pyg_data(snap), store_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["serve", "--model", str(model_path), "--store", str(store_dir)])
    app = serve_cli.build_app_from_argv()
    client = TestClient(app)

    response = client.get("/risk/no-database-url-run")

    assert response.status_code == 401
