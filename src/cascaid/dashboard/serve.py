"""Dashboard API CLI (PRD 5.2):

    python -m cascaid.dashboard.serve --database-url ... [--store data/graph_store]
        [--host 0.0.0.0] [--port 8001]

Serves the risk graph and track record from the Graph Store + Storage to the
frontend, Grafana panel, and MCP server (Section 4.7).
"""

from __future__ import annotations

import argparse

import uvicorn

from cascaid.dashboard.api import create_app
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", type=str, required=True)
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args(argv)


def build_app(args: argparse.Namespace):
    init_db(get_engine(args.database_url))
    return create_app(store_dir=args.store, session_factory=make_session_factory(args.database_url))


def build_app_from_argv(argv: list[str] | None = None):
    return build_app(parse_args(argv))


def main():
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
