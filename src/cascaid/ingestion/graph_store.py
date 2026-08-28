"""Persists PyG graph snapshots to disk, versioned by timestamp (PRD 5.2 Graph Store).

No dedicated graph database is needed for MVP -- a Data object built by
snapshot_builder.to_pyg_data() is torch.save()'d under store_dir/<run_id>/, and the
Model Serving API loads the latest one per run/pipeline at inference time.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


def save_snapshot(data: Data, store_dir: str | Path) -> Path:
    run_dir = Path(store_dir) / data.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{data.step}.pt"
    torch.save(data, path)
    return path


def load_snapshot(path: str | Path) -> Data:
    return torch.load(path, weights_only=False)


def list_snapshots(store_dir: str | Path, run_id: str) -> list[Path]:
    run_dir = Path(store_dir) / run_id
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("*.pt"), key=lambda p: int(p.stem))


def latest_snapshot(store_dir: str | Path, run_id: str) -> Data | None:
    paths = list_snapshots(store_dir, run_id)
    if not paths:
        return None
    return load_snapshot(paths[-1])
