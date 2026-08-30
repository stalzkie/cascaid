"""End-to-end train/eval CLI (PRD Phase 1 steps 3-5):

    python -m cascaid.train [--data data/runs] [--epochs 30] [--out models/pretrained_base.pt]

Builds topology-graph snapshots from the demo pipeline's raw event logs, trains the
GATConv/GINEConv GNN against a flattened XGBoost baseline and a shuffled-adjacency
ablation, reports PR-AUC / lead-time accuracy for all three, and saves the trained GNN
as the pretrained base model (PRD 4.3).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cascaid.benchmarking import save_benchmark
from cascaid.ingestion.labeling import EPICENTER, label_step
from cascaid.ingestion.schema import FEATURE_NAMES, NODE_TYPE_ORDER, NUM_FEATURES
from cascaid.ingestion.snapshot_builder import build_snapshots, shuffle_edge_index, to_pyg_data
from cascaid.ingestion.topology import build_static_graph, load_manifest, load_run_events, load_topology
from cascaid.metrics import RunTrace, lead_time_accuracy, pr_auc
from cascaid.models.baseline import predict_baseline, train_baseline
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.drift import compute_reference, save_reference

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)
RISK_THRESHOLD = 0.5


def build_dataset(data_dir: Path):
    nodes, edges = load_topology(data_dir / "topology.json")
    static_graph = build_static_graph(nodes, edges)
    manifest = load_manifest(data_dir / "manifest.jsonl")

    data_list, shuffled_list = [], []
    rng = np.random.default_rng(123)
    for meta in manifest:
        run_path = data_dir / meta["scenario"] / f"{meta['run_id']}.jsonl"
        events = load_run_events(run_path)
        snapshots = build_snapshots(nodes, edges, events)
        for snap in snapshots:
            labels, usable = label_step(
                meta["scenario"],
                snap.step,
                snap.node_order,
                static_graph,
                meta["fault_onset_step"],
                meta["cascade_step"],
            )
            d = to_pyg_data(snap, labels=labels, usable=usable)
            data_list.append(d)

            shuf_edge_index = shuffle_edge_index(snap.edge_index, len(snap.node_order), rng)
            d_shuf = to_pyg_data(snap, labels=labels, usable=usable, edge_index_override=shuf_edge_index)
            shuffled_list.append(d_shuf)

    return data_list, shuffled_list, nodes, edges, manifest


def split_run_ids(manifest: list[dict], val_frac: float = 0.2, seed: int = 0) -> tuple[set[str], set[str]]:
    run_ids = sorted(m["run_id"] for m in manifest)
    rng = np.random.default_rng(seed)
    rng.shuffle(run_ids)
    n_val = max(1, int(len(run_ids) * val_frac))
    val_ids = set(run_ids[:n_val])
    train_ids = set(run_ids[n_val:])
    return train_ids, val_ids


def train_gnn(train_data: list, epochs: int, hidden: int = 32, lr: float = 1e-3) -> CascadeGNN:
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)

    y_all = torch.cat([d.y for d in train_data])
    usable_all = torch.cat([d.usable for d in train_data])
    pos = y_all[usable_all].sum().item()
    neg = usable_all.sum().item() - pos
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss_per_node = loss_fn(logits, batch.y)
            mask = batch.usable.float()
            loss = (loss_per_node * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.4f}")
    return model


def eval_gnn(model: CascadeGNN, data_list: list):
    model.eval()
    y_true, y_score = [], []
    node_scores: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with torch.no_grad():
        for d in data_list:
            logits = model(d.x, d.edge_index, d.edge_attr)
            probs = torch.sigmoid(logits).numpy()
            usable = d.usable.numpy()
            y = d.y.numpy()
            y_true.extend(y[usable].tolist())
            y_score.extend(probs[usable].tolist())
            node_scores[d.run_id][d.step] = dict(zip(d.node_order, probs.tolist()))
    return np.array(y_true), np.array(y_score), node_scores


def build_traces(manifest: list[dict], run_ids: set[str], node_scores: dict) -> list[RunTrace]:
    traces = []
    for meta in manifest:
        if meta["run_id"] not in run_ids or meta["fault_onset_step"] is None:
            continue
        epicenters = EPICENTER[meta["scenario"]]
        steps_scores = node_scores.get(meta["run_id"], {})
        steps = sorted(steps_scores.keys())
        scores = [max(steps_scores[s].get(e, 0.0) for e in epicenters) for s in steps]
        traces.append(
            RunTrace(
                run_id=meta["run_id"],
                fault_onset_step=meta["fault_onset_step"],
                cascade_step=meta["cascade_step"],
                steps=steps,
                scores=scores,
            )
        )
    return traces


def baseline_node_scores(model, data_list: list) -> dict:
    node_scores: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    y_true, y_score = [], []
    for d in data_list:
        x = d.x.numpy()
        probs = predict_baseline(model, x)
        usable = d.usable.numpy()
        y = d.y.numpy()
        y_true.extend(y[usable].tolist())
        y_score.extend(probs[usable].tolist())
        node_scores[d.run_id][d.step] = dict(zip(d.node_order, probs.tolist()))
    return np.array(y_true), np.array(y_score), node_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", type=str, default="models/pretrained_base.pt")
    parser.add_argument("--benchmarks", type=str, default="benchmarks")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Controls model init, minibatch order, and the train/val split -- "
        "same seed + same data reproduces the same trained model.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    print(f"Loading + building snapshots from {data_dir} ...")
    data_list, shuffled_list, nodes, edges, manifest = build_dataset(data_dir)
    print(f"Built {len(data_list)} graph snapshots across {len(manifest)} runs")

    train_ids, val_ids = split_run_ids(manifest, seed=args.seed)
    train_data = [d for d in data_list if d.run_id in train_ids]
    val_data = [d for d in data_list if d.run_id in val_ids]
    shuf_train_data = [d for d in shuffled_list if d.run_id in train_ids]
    shuf_val_data = [d for d in shuffled_list if d.run_id in val_ids]

    print(f"Train snapshots: {len(train_data)}  Val snapshots: {len(val_data)}")

    print("\n[1/3] Training GNN (real adjacency)...")
    gnn = train_gnn(train_data, epochs=args.epochs)
    y_true_gnn, y_score_gnn, node_scores_gnn = eval_gnn(gnn, val_data)

    print("\n[2/3] Training GNN (shuffled-adjacency ablation)...")
    gnn_shuf = train_gnn(shuf_train_data, epochs=args.epochs)
    y_true_shuf, y_score_shuf, node_scores_shuf = eval_gnn(gnn_shuf, shuf_val_data)

    print("\n[3/3] Training flattened XGBoost baseline...")
    x_train = np.stack([d.x.numpy() for d in train_data]).reshape(-1, IN_DIM)
    y_train_flat = np.concatenate([d.y.numpy() for d in train_data])
    usable_train_flat = np.concatenate([d.usable.numpy() for d in train_data])
    baseline = train_baseline(x_train[usable_train_flat], y_train_flat[usable_train_flat])
    y_true_base, y_score_base, node_scores_base = baseline_node_scores(baseline, val_data)

    gnn_trace = build_traces(manifest, val_ids, node_scores_gnn)
    shuf_trace = build_traces(manifest, val_ids, node_scores_shuf)
    base_trace = build_traces(manifest, val_ids, node_scores_base)

    results = {
        "GNN (real adjacency)": (
            pr_auc(y_true_gnn, y_score_gnn),
            lead_time_accuracy(gnn_trace, RISK_THRESHOLD),
        ),
        "GNN (shuffled adjacency, ablation)": (
            pr_auc(y_true_shuf, y_score_shuf),
            lead_time_accuracy(shuf_trace, RISK_THRESHOLD),
        ),
        "XGBoost (flattened baseline)": (
            pr_auc(y_true_base, y_score_base),
            lead_time_accuracy(base_trace, RISK_THRESHOLD),
        ),
    }

    print("\n=== Results (validation set) ===")
    print(f"{'Model':<36}{'PR-AUC':>10}{'Detect rate':>14}{'Mean lead (steps)':>20}")
    for name, (auc, lt) in results.items():
        print(f"{name:<36}{auc:>10.3f}{lt['detection_rate']:>14.2f}{lt['mean_lead_time_steps']:>20.2f}")

    benchmark_dir = save_benchmark(results, Path(args.benchmarks))
    print(f"\nSaved benchmark comparison chart to {benchmark_dir / 'comparison.png'}")
    print(f"Previous run (if any) archived under {Path(args.benchmarks) / 'archive'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gnn.state_dict(), out_path)
    print(f"\nSaved pretrained GNN to {out_path}")

    # Reference distribution for drift detection (PRD 7) -- computed once here so
    # serving never needs the raw training data, just this. Only the continuous
    # features (x_train's first NUM_FEATURES columns); the one-hot node-type
    # columns that follow aren't a distribution PSI is meaningful over.
    reference = compute_reference(x_train[:, :NUM_FEATURES], FEATURE_NAMES)
    reference_path = out_path.with_suffix(".drift_reference.json")
    save_reference(reference, reference_path)
    print(f"Saved drift reference to {reference_path}")


if __name__ == "__main__":
    main()
