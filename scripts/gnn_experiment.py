"""Repeated-seed evaluation harness for the cascade-risk GNN (see
docs/GNN_Accuracy_Improvement_Log.md). Reuses cascaid.train's building blocks
directly rather than duplicating them -- this script only adds the seeding
and repetition train.py's CLI doesn't do.

    python scripts/gnn_experiment.py --data data/runs [--seeds 5] [--epochs 30]
        [--hidden 32] [--lr 0.001] [--layers 2] [--conv gine]

Trains the GNN `--seeds` times against the SAME generated data (fixing model
init + train/val split + minibatch order per seed), reporting mean/std
PR-AUC and detection rate so a hyperparameter change can be judged against
the noise floor instead of a single noisy run.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cascaid.ingestion.schema import NUM_FEATURES
from cascaid.metrics import brier_score, expected_calibration_error, lead_time_accuracy, pr_auc
from cascaid.models.gnn import CascadeGNN
from cascaid.train import IN_DIM, build_dataset, build_traces, eval_gnn, split_run_ids


def train_gnn_seeded(
    train_data: list, epochs: int, hidden: int, lr: float, layers: int, conv: str, weight_decay: float = 0.0
) -> CascadeGNN:
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=hidden, layers=layers, conv=conv)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)

    y_all = torch.cat([d.y for d in train_data])
    usable_all = torch.cat([d.usable for d in train_data])
    pos = y_all[usable_all].sum().item()
    neg = usable_all.sum().item() - pos
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss_per_node = loss_fn(logits, batch.y)
            mask = batch.usable.float()
            loss = (loss_per_node * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
    return model


def run_one_seed(
    seed: int,
    data_list: list,
    manifest: list[dict],
    epochs: int,
    hidden: int,
    lr: float,
    layers: int,
    conv: str,
    weight_decay: float = 0.0,
) -> dict:
    torch.manual_seed(seed)
    train_ids, val_ids = split_run_ids(manifest, seed=seed)
    train_data = [d for d in data_list if d.run_id in train_ids]
    val_data = [d for d in data_list if d.run_id in val_ids]

    model = train_gnn_seeded(
        train_data, epochs=epochs, hidden=hidden, lr=lr, layers=layers, conv=conv, weight_decay=weight_decay
    )
    y_true, y_score, node_scores = eval_gnn(model, val_data)

    auc = pr_auc(y_true, y_score)
    lead = lead_time_accuracy(build_traces(manifest, val_ids, node_scores), threshold=0.5)
    return {
        "pr_auc": auc,
        "detection_rate": lead["detection_rate"],
        "mean_lead": lead["mean_lead_time_steps"],
        "brier": brier_score(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/runs")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv", type=str, default="gine", choices=["gine", "gat"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    data_dir = Path(args.data)
    print(f"Loading + building snapshots from {data_dir} ...")
    data_list, _shuffled_list, _nodes, _edges, manifest = build_dataset(data_dir)
    print(f"Built {len(data_list)} graph snapshots across {len(manifest)} runs")
    print(
        f"Config: seeds={args.seeds} epochs={args.epochs} hidden={args.hidden} "
        f"lr={args.lr} layers={args.layers} conv={args.conv}\n"
    )

    results = []
    for seed in range(args.seeds):
        r = run_one_seed(
            seed, data_list, manifest, args.epochs, args.hidden, args.lr, args.layers, args.conv, args.weight_decay
        )
        print(
            f"  seed={seed}  PR-AUC={r['pr_auc']:.3f}  detect={r['detection_rate']:.2f}  "
            f"lead={r['mean_lead']:.2f}  brier={r['brier']:.3f}  ECE={r['ece']:.3f}"
        )
        results.append(r)

    aucs = [r["pr_auc"] for r in results if not np.isnan(r["pr_auc"])]
    detects = [r["detection_rate"] for r in results if not np.isnan(r["detection_rate"])]
    briers = [r["brier"] for r in results if not np.isnan(r["brier"])]
    eces = [r["ece"] for r in results if not np.isnan(r["ece"])]
    print("\n=== Summary over {} seeds ===".format(args.seeds))
    print(
        f"PR-AUC:          mean={statistics.mean(aucs):.3f}  std={statistics.pstdev(aucs):.3f}  "
        f"min={min(aucs):.3f}  max={max(aucs):.3f}"
    )
    print(f"Detection rate:  mean={statistics.mean(detects):.3f}  std={statistics.pstdev(detects):.3f}")
    print(
        f"Brier score:     mean={statistics.mean(briers):.3f}  std={statistics.pstdev(briers):.3f}  "
        "(0=perfectly calibrated, 0.25=uninformative)"
    )
    print(
        f"ECE:             mean={statistics.mean(eces):.3f}  std={statistics.pstdev(eces):.3f}  "
        "(mean gap between predicted confidence and empirical accuracy per bucket)"
    )


if __name__ == "__main__":
    main()
