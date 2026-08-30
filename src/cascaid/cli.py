"""Unified `cascaid` CLI (Auto-Instrumentation Glue Layer Plan, step 1): one binary
dispatching to the existing entry points instead of separate `python -m cascaid.xxx`
invocations. `serve`/`train`/`dashboard` are thin passthroughs -- each rewrites
sys.argv and calls the target module's own main(), the exact seam the existing e2e
tests already drive directly. `demo` is new: it orchestrates the run_scenarios ->
train -> seed_store sequence docker-compose's seed service already runs, so a beta
tester gets the same local demo experience without Docker.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cascaid.auth.configure as auth_configure_cli
import cascaid.dashboard.serve as dashboard_cli
import cascaid.drift as drift_cli
import cascaid.import_langfuse as import_langfuse_cli
import cascaid.ingest as ingest_cli
import cascaid.mcp.server as mcp_server_cli
import cascaid.serve as serve_cli
import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
import cascaid_demo.seed_store as seed_store_cli

SUBCOMMANDS = ("serve", "train", "dashboard", "demo", "run", "ingest", "auth", "mcp", "drift", "import")

_SITECUSTOMIZE_SOURCE = "from cascaid._instrument_bootstrap import bootstrap\nbootstrap()\n"


def _delegate(prog: str, module_main, rest: list[str]) -> None:
    sys.argv = [prog, *rest]
    module_main()


def _run_demo(rest: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="cascaid demo")
    parser.add_argument("--runs-per-scenario", type=int, default=25)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--data", type=str, default="data/runs")
    parser.add_argument("--model", type=str, default="models/pretrained_base.pt")
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Defaults to a local SQLite file under --store's parent dir -- no "
        "Postgres required for the local demo (PRD 4.2).",
    )
    args = parser.parse_args(rest)

    database_url = args.database_url or f"sqlite:///{Path(args.store).parent / 'cascaid-demo.db'}"

    _delegate(
        "run_scenarios",
        run_scenarios_cli.main,
        ["--runs-per-scenario", str(args.runs_per_scenario), "--steps", str(args.steps), "--out", args.data],
    )
    _delegate(
        "train",
        train_cli.main,
        ["--data", args.data, "--epochs", str(args.epochs), "--out", args.model],
    )
    _delegate(
        "seed_store",
        seed_store_cli.main,
        ["--data", args.data, "--model", args.model, "--store", args.store, "--database-url", database_url],
    )
    print(f"\nDemo ready -- database: {database_url}  graph store: {args.store}")


def _run_instrumented(rest: list[str]) -> None:
    """`cascaid run -- <command>`: launches the customer's own pipeline as a
    subprocess with instrumentation already applied before its code runs, using
    the same sitecustomize-injection trick ddtrace-run uses (a directory holding a
    tiny sitecustomize.py prepended to PYTHONPATH -- Python's site module imports
    it automatically at interpreter startup, ahead of anything the target command
    does). This is what makes zero-code-change instrumentation (PRD 4.1) real: no
    wrapper the customer has to call, no line they have to add to their own app.
    """
    parser = argparse.ArgumentParser(prog="cascaid run")
    parser.add_argument("--events", type=str, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(rest)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("usage: cascaid run [--events PATH] -- <command> [args...]", file=sys.stderr)
        raise SystemExit(2)

    run_id = str(uuid.uuid4())
    events_path = args.events or f"data/live/{run_id}.jsonl"
    Path(events_path).parent.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = Path(tempfile.mkdtemp(prefix="cascaid-sitecustomize-"))
    (bootstrap_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE_SOURCE, encoding="utf-8")

    env = dict(os.environ)
    env["CASCAID_RUN_ID"] = run_id
    env["CASCAID_EVENTS_PATH"] = events_path
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(bootstrap_dir) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    print(f"cascaid run: instrumenting `{' '.join(command)}` (run_id={run_id})")
    print(f"cascaid run: events -> {events_path}")
    result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] not in SUBCOMMANDS:
        print(f"usage: cascaid {{{','.join(SUBCOMMANDS)}}} ...", file=sys.stderr)
        raise SystemExit(2)

    subcommand, rest = argv[0], argv[1:]
    if subcommand == "train":
        _delegate("train", train_cli.main, rest)
    elif subcommand == "serve":
        _delegate("serve", serve_cli.main, rest)
    elif subcommand == "dashboard":
        _delegate("dashboard", dashboard_cli.main, rest)
    elif subcommand == "demo":
        _run_demo(rest)
    elif subcommand == "run":
        _run_instrumented(rest)
    elif subcommand == "ingest":
        _delegate("ingest", ingest_cli.main, rest)
    elif subcommand == "auth":
        if not rest or rest[0] != "configure":
            print("usage: cascaid auth configure ...", file=sys.stderr)
            raise SystemExit(2)
        _delegate("auth configure", auth_configure_cli.main, rest[1:])
    elif subcommand == "mcp":
        _delegate("mcp", mcp_server_cli.main, rest)
    elif subcommand == "drift":
        _delegate("drift", drift_cli.main, rest)
    elif subcommand == "import":
        if not rest or rest[0] != "langfuse":
            print("usage: cascaid import langfuse ...", file=sys.stderr)
            raise SystemExit(2)
        _delegate("import langfuse", import_langfuse_cli.main, rest[1:])


if __name__ == "__main__":
    main()
