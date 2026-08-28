"""Live Ingestion CLI (Auto-Instrumentation Glue Layer Plan, closing the
persistence gap): reads the JSON-lines event log `cascaid run` writes
(CASCAID_EVENTS_PATH) and turns it into Graph Store snapshots + score history,
reusing exactly the functions seed_store.py already uses for demo data
(build_snapshots, to_pyg_data, save_snapshot, predict_risk, record_scores) --
just fed from a live event log instead of a synthetic run corpus. Runs as its
own process, deliberately separate from the customer's instrumented app (which
`cascaid run` keeps lightweight): loading torch/the model here, not inside their
live request path, is the same reasoning that kept register_litellm_callbacks
etc. defensive against breaking a customer's process.

    python -m cascaid.ingest --events data/live/<run_id>.jsonl --store data/graph_store
        [--model models/pretrained_base.pt --database-url ...] [--follow]

Snapshot building recomputes from the full accumulated event history on every
call (matching build_snapshots' own rolling-window semantics exactly, the same
window a demo-trained model was validated against) rather than a separate
incremental algorithm -- simpler and provably identical to the batch path, at
the cost of re-processing older events on each --follow tick. Fine for a beta
pass; revisit if a long-running session's event volume makes that costly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import NODE_TYPE_ORDER, NUM_FEATURES, CallEvent, NodeType
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.serving.risk import load_model, predict_risk
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db, record_scores

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)


def read_events_file(events_path: str | Path) -> tuple[dict[str, NodeType], list[tuple[str, str]], list[CallEvent]]:
    """The last topology record wins (one compiled pipeline per `cascaid run`,
    matching instrument_langgraph's single-global-step-counter simplification)."""
    nodes: dict[str, NodeType] = {}
    edges: list[tuple[str, str]] = []
    events: list[CallEvent] = []

    path = Path(events_path)
    if not path.exists():
        return nodes, edges, events

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["type"] == "topology":
            nodes = {name: NodeType(value) for name, value in record["nodes"].items()}
            edges = [tuple(edge) for edge in record["edges"]]
        elif record["type"] == "call_event":
            events.append(CallEvent.from_json({k: v for k, v in record.items() if k != "type"}))

    return nodes, edges, events


def ingest_once(
    events_path: str | Path,
    store_dir: str | Path,
    model=None,
    session_factory=None,
    hidden: int = 32,
) -> dict:
    nodes, edges, events = read_events_file(events_path)
    if not nodes or not events:
        return {"snapshots": 0, "run_id": None}

    snapshot_count = 0
    run_id = None
    with (session_factory() if session_factory else _null_session()) as session:
        for snapshot in build_snapshots(nodes, edges, events):
            data = to_pyg_data(snapshot)
            save_snapshot(data, store_dir)
            run_id = data.run_id
            if model is not None and session is not None:
                scores = predict_risk(model, data)
                record_scores(session, run_id=data.run_id, step=data.step, scores=scores)
            snapshot_count += 1

    return {"snapshots": snapshot_count, "run_id": run_id}


class _null_session:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--database-url", type=str, default=None)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=3.0)
    return parser.parse_args(argv)


def main():
    args = parse_args()

    model = None
    if args.model:
        model = load_model(args.model, in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=args.hidden)

    session_factory = None
    if args.database_url:
        init_db(get_engine(args.database_url))
        session_factory = make_session_factory(args.database_url)

    result = ingest_once(args.events, args.store, model=model, session_factory=session_factory, hidden=args.hidden)
    print(f"Ingested {result['snapshots']} snapshot(s) for run_id={result['run_id']}")

    while args.follow:
        time.sleep(args.interval)
        result = ingest_once(args.events, args.store, model=model, session_factory=session_factory, hidden=args.hidden)
        print(f"Ingested {result['snapshots']} snapshot(s) for run_id={result['run_id']}")


if __name__ == "__main__":
    main()
