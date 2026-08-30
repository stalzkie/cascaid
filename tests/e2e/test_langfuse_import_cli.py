"""E2E seam: the real `cascaid import langfuse` CLI entry point, argv to a real
sqlite DB on disk."""

import json
import sys

import pytest

import cascaid.langfuse_import as import_langfuse_cli
from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import get_incidents


@pytest.mark.e2e
def test_import_langfuse_cli_records_incidents_from_a_real_scores_file(tmp_path, monkeypatch, capsys):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            [
                {"name": "helpfulness", "value": 0.95, "dataType": "NUMERIC", "timestamp": "2026-08-01T00:00:00Z"},
                {"name": "toxicity", "value": 0.05, "dataType": "NUMERIC", "timestamp": "2026-08-02T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import-langfuse",
            "--file",
            str(scores_path),
            "--database-url",
            database_url,
            "--run-id",
            "run-1",
            "--node-name",
            "agent-checkout",
            "--threshold",
            "0.5",
        ],
    )

    import_langfuse_cli.main()

    output = capsys.readouterr().out
    assert "Imported 1 incident(s) from 2 score(s)" in output

    with make_session_factory(database_url)() as session:
        incidents = get_incidents(session, run_id="run-1")
    assert len(incidents) == 1
    assert incidents[0].node_name == "agent-checkout"
