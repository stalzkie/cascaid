"""Model Serving CLI (PRD 5.2):

    python -m cascaid.serve [--model models/pretrained_base.pt] [--store data/graph_store]
        [--hidden 32] [--host 0.0.0.0] [--port 8000]

Loads the pretrained/fine-tuned GNN and serves per-node cascade risk scores for the
latest graph snapshot of a given run_id, read from the Graph Store.
"""

from __future__ import annotations

import argparse

import uvicorn

from cascaid.ingestion.schema import NODE_TYPE_ORDER, NUM_FEATURES
from cascaid.serving.api import create_app
from cascaid.serving.risk import load_model
from cascaid.storage.db import make_session_factory

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pretrained_base.pt")
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="If set, persists every served risk score to this Postgres/SQLAlchemy URL (PRD 5.2 Storage).",
    )
    return parser.parse_args(argv)


def build_app(args: argparse.Namespace):
    model = load_model(args.model, in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=args.hidden)
    session_factory = make_session_factory(args.database_url) if args.database_url else None
    return create_app(model=model, store_dir=args.store, session_factory=session_factory)


def build_app_from_argv(argv: list[str] | None = None):
    return build_app(parse_args(argv))


def main():
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
