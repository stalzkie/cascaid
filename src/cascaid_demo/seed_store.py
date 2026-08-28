"""Seeds the Graph Store + score history from demo-run data (PRD 4.2 local demo
mode). run_scenarios/train produce raw event logs and a trained model, but nothing
persists a snapshot a live Model Serving API could read -- without this, a fresh
`docker compose up` trains a model and then has nothing to show in the dashboard.
This is the batch equivalent of what a live cascaid.serve /risk call does per
request, run once at startup over every demo run so the dashboard has data
immediately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import NODE_TYPE_ORDER, NUM_FEATURES
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.ingestion.topology import load_manifest, load_run_events, load_topology
from cascaid.serving.risk import load_model, predict_risk
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import init_db, record_scores

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)


def seed(
    data_dir: str | Path,
    model_path: str | Path,
    store_dir: str | Path,
    session_factory: sessionmaker[Session],
    hidden: int = 32,
) -> dict:
    data_dir = Path(data_dir)
    nodes, edges = load_topology(data_dir / "topology.json")
    manifest = load_manifest(data_dir / "manifest.jsonl")
    model = load_model(model_path, in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=hidden)

    run_ids: set[str] = set()
    snapshot_count = 0
    with session_factory() as session:
        for meta in manifest:
            run_path = data_dir / meta["scenario"] / f"{meta['run_id']}.jsonl"
            events = load_run_events(run_path)
            for snapshot in build_snapshots(nodes, edges, events):
                data = to_pyg_data(snapshot)
                save_snapshot(data, store_dir)
                scores = predict_risk(model, data)
                record_scores(session, run_id=meta["run_id"], step=data.step, scores=scores)
                snapshot_count += 1
            run_ids.add(meta["run_id"])

    return {"runs": len(run_ids), "snapshots": snapshot_count, "run_ids": run_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/runs")
    parser.add_argument("--model", type=str, default="models/pretrained_base.pt")
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--database-url", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    args = parser.parse_args()

    init_db(get_engine(args.database_url))
    counts = seed(
        data_dir=args.data,
        model_path=args.model,
        store_dir=args.store,
        session_factory=make_session_factory(args.database_url),
        hidden=args.hidden,
    )
    print(f"Seeded {counts['snapshots']} snapshots across {counts['runs']} demo runs into the graph store + scores.")


if __name__ == "__main__":
    main()
