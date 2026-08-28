"""E2E seam: `cascaid demo`, wired end-to-end through the real CLI entry points --
the same run_scenarios -> train -> seed_store sequence docker-compose's seed
service runs, but reachable as one call with no Postgres required (Auto-
Instrumentation Glue Layer Plan, step 1: `cascaid demo` must give value with
zero infra connected, PRD 4.2)."""

from __future__ import annotations

import sys

import pytest

import cascaid.cli as cli
from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import get_score_history


@pytest.mark.e2e
def test_demo_seeds_a_local_store_without_a_database_url(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"
    store_dir = tmp_path / "graph_store"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cascaid",
            "demo",
            "--runs-per-scenario",
            "2",
            "--steps",
            "15",
            "--epochs",
            "2",
            "--data",
            str(data_dir),
            "--model",
            str(model_path),
            "--store",
            str(store_dir),
        ],
    )

    cli.main(sys.argv[1:])

    assert model_path.exists()

    printed = capsys.readouterr().out
    database_url = next(
        (line.split("database: ", 1)[1].split()[0] for line in printed.splitlines() if "database: " in line),
        None,
    )
    assert database_url is not None and database_url.startswith("sqlite:///")

    manifest_lines = (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert manifest_lines  # sanity: run_scenarios actually wrote a manifest
    import json

    first_run_id = json.loads(manifest_lines[0])["run_id"]

    with make_session_factory(database_url)() as session:
        history = get_score_history(session, run_id=first_run_id)
    assert len(history) > 0
