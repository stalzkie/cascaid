"""Retrains the cascade-risk GNN on a live install's own accumulated data --
Graph Store snapshots plus IncidentLabel rows -- instead of the synthetic
fault-injection scenarios cascaid.train uses (PRD 4.3: "fine-tunes quietly as
real customer data accumulates"). See docs/Real_Data_Retraining_Plan.md for
the design this implements, including why real-data labeling is node-local
only (label_step_from_incidents), unlike the synthetic path's predecessor
propagation.

    python -m cascaid.retrain --database-url ... --store data/graph_store \
        --out models/pretrained_base.pt [--min-pr-auc 0.7] [--epochs 60]
        [--hidden 32] [--layers 2] [--conv gine]

Only overwrites --out if the newly trained model's PR-AUC on a held-out
split of the real runs is at or above --min-pr-auc -- a small, noisy batch of
real incidents must not be able to silently make the served model worse than
whatever it's replacing. Manual/cron-triggered for now; wiring this to fire
automatically off cascaid.drift's existing PSI check is a natural fast-follow
once this command is proven, not a prerequisite for it.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from cascaid.ingestion.graph_store import list_runs, list_snapshots, load_snapshot
from cascaid.ingestion.labeling import label_step_from_incidents
from cascaid.ingestion.schema import NODE_TYPE_ORDER, NUM_FEATURES
from cascaid.metrics import brier_score, expected_calibration_error, pr_auc
from cascaid.models.gnn import CascadeGNN
from cascaid.models.model_config import ModelConfig, save_model_config
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_incidents, init_db
from cascaid.train import eval_gnn, split_run_ids, train_gnn

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)


def build_real_dataset(
    store_dir: str | Path, run_ids: list[str], incidents_by_run: dict[str, list[tuple[str, datetime]]]
) -> list:
    data_list = []
    for run_id in run_ids:
        incidents = incidents_by_run.get(run_id, [])
        for path in list_snapshots(store_dir, run_id):
            data = load_snapshot(path)
            labels, usable = label_step_from_incidents(data.node_order, incidents, data.step_start, data.step_end)
            data.y = torch.tensor([labels[n] for n in data.node_order], dtype=torch.float32)
            data.usable = torch.tensor([usable[n] for n in data.node_order], dtype=torch.bool)
            data_list.append(data)
    return data_list


def retrain(
    store_dir: str | Path,
    incidents_by_run: dict[str, list[tuple[str, datetime]]],
    epochs: int = 60,
    seed: int = 0,
    hidden: int = 32,
    layers: int = 2,
    conv: str = "gine",
) -> tuple[CascadeGNN, dict]:
    run_ids = list_runs(store_dir)
    if not run_ids:
        raise ValueError(f"No runs found in Graph Store at {store_dir!r}")

    data_list = build_real_dataset(store_dir, run_ids, incidents_by_run)
    if not data_list:
        raise ValueError("No snapshots found for any known run")

    train_ids, val_ids = split_run_ids([{"run_id": r} for r in run_ids], seed=seed)
    train_data = [d for d in data_list if d.run_id in train_ids]
    val_data = [d for d in data_list if d.run_id in val_ids]

    torch.manual_seed(seed)
    model = train_gnn(train_data, epochs=epochs, hidden=hidden, layers=layers, conv=conv)
    y_true, y_score, _ = eval_gnn(model, val_data)

    metrics = {
        "pr_auc": pr_auc(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
        "num_train_snapshots": len(train_data),
        "num_val_snapshots": len(val_data),
    }
    return model, metrics


def should_swap_model(pr_auc_value: float, min_pr_auc: float) -> bool:
    """A NaN PR-AUC (the held-out split had only one class) never swaps --
    there's no evidence the new model is any good, so the safe default is to
    keep serving whatever's already there."""
    if math.isnan(pr_auc_value):
        return False
    return pr_auc_value >= min_pr_auc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", type=str, required=True)
    parser.add_argument("--store", type=str, default="data/graph_store")
    parser.add_argument("--out", type=str, default="models/pretrained_base.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pr-auc", type=float, default=0.7)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv", type=str, default="gine", choices=["gine", "gat"])
    return parser.parse_args(argv)


def main():
    args = parse_args()
    init_db(get_engine(args.database_url))
    session_factory = make_session_factory(args.database_url)
    with session_factory() as session:
        run_ids = list_runs(args.store)

        # SQLite (unlike Postgres) doesn't preserve tzinfo on a DateTime(timezone=True)
        # column round trip -- same fix as repository.py's session-expiry check.
        def _as_utc(occurred_at: datetime) -> datetime:
            return occurred_at if occurred_at.tzinfo else occurred_at.replace(tzinfo=timezone.utc)

        incidents_by_run = {
            run_id: [
                (incident.node_name, _as_utc(incident.occurred_at))
                for incident in get_incidents(session, run_id=run_id)
            ]
            for run_id in run_ids
        }

    model, metrics = retrain(
        args.store,
        incidents_by_run,
        epochs=args.epochs,
        seed=args.seed,
        hidden=args.hidden,
        layers=args.layers,
        conv=args.conv,
    )

    print(
        f"Real-data retrain: PR-AUC={metrics['pr_auc']:.3f}  Brier={metrics['brier']:.3f}  "
        f"ECE={metrics['ece']:.3f}  train={metrics['num_train_snapshots']}  val={metrics['num_val_snapshots']}"
    )

    if not should_swap_model(metrics["pr_auc"], args.min_pr_auc):
        print(f"PR-AUC did not clear --min-pr-auc {args.min_pr_auc} -- keeping the existing model at {args.out}")
        return

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp_path)
    tmp_path.replace(out_path)
    save_model_config(
        ModelConfig(in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=args.hidden, layers=args.layers, conv=args.conv),
        out_path,
    )
    print(f"Swapped in the newly retrained model at {out_path}")


if __name__ == "__main__":
    main()
