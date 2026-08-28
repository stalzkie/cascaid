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
import sys
from pathlib import Path

import cascaid.dashboard.serve as dashboard_cli
import cascaid.serve as serve_cli
import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli
import cascaid_demo.seed_store as seed_store_cli

SUBCOMMANDS = ("serve", "train", "dashboard", "demo")


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


if __name__ == "__main__":
    main()
