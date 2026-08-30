"""Unit seam: cascaid.cli's subcommand dispatch (Auto-Instrumentation Glue Layer
Plan, step 1). Delegate targets are monkeypatched to spies so these stay fast/no-I/O
-- the real end-to-end behavior of each delegated command is already covered by its
own module's e2e tests (test_serve_cli.py etc.); this file only proves cli.main()
routes argv to the right one."""

from __future__ import annotations

from pathlib import Path

import pytest

import cascaid.cli as cli


def test_main_exits_nonzero_with_no_subcommand():
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code != 0


def test_main_exits_nonzero_with_an_unknown_subcommand():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["not-a-real-subcommand"])
    assert exc_info.value.code != 0


def test_main_dispatches_train_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.train_cli, "main", fake_main)

    cli.main(["train", "--data", "data/runs", "--epochs", "3"])

    assert captured["argv"] == ["train", "--data", "data/runs", "--epochs", "3"]


def test_main_dispatches_serve_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.serve_cli, "main", fake_main)

    cli.main(["serve", "--model", "models/pretrained_base.pt"])

    assert captured["argv"] == ["serve", "--model", "models/pretrained_base.pt"]


def test_main_dispatches_dashboard_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.dashboard_cli, "main", fake_main)

    cli.main(["dashboard", "--store", "data/graph_store"])

    assert captured["argv"] == ["dashboard", "--store", "data/graph_store"]


def test_main_dispatches_ingest_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.ingest_cli, "main", fake_main)

    cli.main(["ingest", "--events", "data/live/run-1.jsonl", "--store", "data/graph_store"])

    assert captured["argv"] == ["ingest", "--events", "data/live/run-1.jsonl", "--store", "data/graph_store"]


def test_main_dispatches_auth_configure_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.auth_configure_cli, "main", fake_main)

    cli.main(["auth", "configure", "--database-url", "sqlite:///x.db", "--set-username", "admin"])

    assert captured["argv"] == ["auth configure", "--database-url", "sqlite:///x.db", "--set-username", "admin"]


def test_main_exits_nonzero_for_an_unknown_auth_subcommand():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["auth", "not-a-real-subcommand"])
    assert exc_info.value.code != 0


def test_main_dispatches_mcp_with_passthrough_args_as_argv(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_main():
        import sys as sys_

        captured["argv"] = list(sys_.argv)

    monkeypatch.setattr(cli.mcp_server_cli, "main", fake_main)

    cli.main(["mcp", "--database-url", "sqlite:///x.db", "--store", "data/graph_store"])

    assert captured["argv"] == ["mcp", "--database-url", "sqlite:///x.db", "--store", "data/graph_store"]


def test_main_dispatches_run_by_launching_the_target_as_an_instrumented_subprocess(monkeypatch, tmp_path):
    captured = {}

    class _FakeCompletedProcess:
        returncode = 0

    def fake_subprocess_run(command, env, **kwargs):
        captured["command"] = command
        captured["env"] = env
        return _FakeCompletedProcess()

    monkeypatch.setattr(cli.subprocess, "run", fake_subprocess_run)

    events_path = tmp_path / "events.jsonl"
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--events", str(events_path), "--", "python", "app.py", "--flag"])

    assert exc_info.value.code == 0
    assert captured["command"] == ["python", "app.py", "--flag"]
    assert captured["env"]["CASCAID_EVENTS_PATH"] == str(events_path)
    assert captured["env"]["CASCAID_RUN_ID"]  # a run_id was generated
    # The generated sitecustomize.py's directory must come first so it's importable
    # before the target command's own code runs.
    bootstrap_dir = captured["env"]["PYTHONPATH"].split(cli.os.pathsep)[0]
    assert (Path(bootstrap_dir) / "sitecustomize.py").exists()
