"""Unit seam: cascaid.cli's subcommand dispatch (Auto-Instrumentation Glue Layer
Plan, step 1). Delegate targets are monkeypatched to spies so these stay fast/no-I/O
-- the real end-to-end behavior of each delegated command is already covered by its
own module's e2e tests (test_serve_cli.py etc.); this file only proves cli.main()
routes argv to the right one."""

from __future__ import annotations

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
