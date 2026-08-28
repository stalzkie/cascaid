"""E2E seam: run_scenarios -> train -> seed_store, wired end-to-end through the
real CLI entry points -- this is the exact sequence docker-compose's seed
service runs on first `docker compose up` (PRD 4.2 local demo mode)."""

import json
import sys

import pytest

import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
import cascaid_demo.seed_store as seed_store_cli
from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import get_score_history


@pytest.mark.e2e
def test_seed_store_cli_makes_demo_data_immediately_visible_to_serve(tmp_path, monkeypatch):
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scenarios", "--runs-per-scenario", "2", "--steps", "15", "--out", str(data_dir), "--seed", "0"],
    )
    run_scenarios_cli.main()

    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data_dir), "--epochs", "2", "--out", str(model_path)])
    train_cli.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seed_store",
            "--data",
            str(data_dir),
            "--model",
            str(model_path),
            "--store",
            str(store_dir),
            "--database-url",
            database_url,
        ],
    )
    seed_store_cli.main()

    manifest_lines = (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert manifest_lines  # sanity: run_scenarios actually wrote a manifest
    first_run_id = json.loads(manifest_lines[0])["run_id"]

    # Score history should already be there without this test ever calling
    # cascaid.serve's /risk endpoint itself -- seed_store did the inference.
    with make_session_factory(database_url)() as session:
        history = get_score_history(session, run_id=first_run_id)
    assert len(history) > 0
