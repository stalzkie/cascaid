"""One-off diagnostic (see docs/GNN_Accuracy_Improvement_Log.md, Finding 3):
trains once at the current best config, then recomputes PR-AUC on the SAME
predicted scores under a stricter positive-label definition (only the later
half of the fault ramp, where fault_progress >= 0.5) to test whether the
~0.77-0.8 ceiling comes from inherently ambiguous early-ramp labels rather
than the model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gnn_experiment import train_gnn_seeded

from cascaid.metrics import pr_auc
from cascaid.train import build_dataset, eval_gnn, split_run_ids

RAMP_STEPS = 10  # matches fault_injection.make_scenario's default


def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs_60")
    seed = 0

    print(f"Loading + building snapshots from {data_dir} ...")
    data_list, _shuf, nodes, edges, manifest = build_dataset(data_dir)
    torch.manual_seed(seed)
    train_ids, val_ids = split_run_ids(manifest, seed=seed)
    train_data = [d for d in data_list if d.run_id in train_ids]
    val_data = [d for d in data_list if d.run_id in val_ids]

    print("Training once at the current best config (epochs=100, hidden=32, gine)...")
    model = train_gnn_seeded(train_data, epochs=100, hidden=32, lr=1e-3, layers=2, conv="gine")
    y_true, y_score, node_scores = eval_gnn(model, val_data)

    original_auc = pr_auc(y_true, y_score)

    # Rebuild y_true/y_score, but for faulty runs' ramp window, only KEEP the
    # step if it's either pre-onset (label must be 0) or in the late half of
    # the ramp (label must be 1, and by now genuinely distinguishable) --
    # dropping the early-ramp steps as ambiguous-by-design instead of
    # counting them against the model.
    meta_by_run = {m["run_id"]: m for m in manifest}
    y_true2, y_score2 = [], []
    dropped = 0
    for d in val_data:
        meta = meta_by_run[d.run_id]
        onset = meta["fault_onset_step"]
        step = d.step
        usable = d.usable.numpy()
        y = d.y.numpy()
        probs = np.array([node_scores[d.run_id][step][n] for n in d.node_order])
        if onset is not None and onset <= step < onset + RAMP_STEPS:
            progress = (step - onset) / RAMP_STEPS
            if progress < 0.5:
                dropped += usable.sum()
                continue
        y_true2.extend(y[usable].tolist())
        y_score2.extend(probs[usable].tolist())

    stricter_auc = pr_auc(np.array(y_true2), np.array(y_score2))

    print(f"\nOriginal PR-AUC (full ramp window counted):     {original_auc:.3f}")
    print(f"Stricter PR-AUC (early-ramp steps dropped):     {stricter_auc:.3f}")
    print(f"(dropped {dropped} early-ramp node-steps as ambiguous-by-design)")


if __name__ == "__main__":
    main()
