"""CLI for importing historical Langfuse quality-degradation scores as Cascaid
incidents (PRD 4: historical incident/degradation labeling, "sourced from
Langfuse/LangSmith/Phoenix exports").

    python -m cascaid.import_langfuse --file scores.json --database-url ... \\
        --run-id <run_id> --node-name <node_name> [--threshold 0.5]
"""

from __future__ import annotations

import argparse

from cascaid.ingestion.langfuse_import import import_langfuse_incidents, parse_langfuse_scores
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--database-url", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--node-name",
        type=str,
        required=True,
        help="Every imported score is attributed to this single Cascaid pipeline "
        "node -- a Langfuse trace has no concept of Cascaid's run_id/node_name, "
        "so there's no automatic mapping to infer it from.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    init_db(get_engine(args.database_url))
    session_factory = make_session_factory(args.database_url)
    scores = parse_langfuse_scores(args.file)
    with session_factory() as session:
        count = import_langfuse_incidents(session, scores, args.run_id, args.node_name, args.threshold)
    print(f"Imported {count} incident(s) from {len(scores)} score(s) in {args.file}")


if __name__ == "__main__":
    main()
