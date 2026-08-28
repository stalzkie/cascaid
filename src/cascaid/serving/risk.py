"""GNN inference over a graph snapshot -> per-node cascade risk score (PRD 5.2
Model Serving). Kept free of FastAPI so it can be unit-tested without HTTP."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from cascaid.models.gnn import CascadeGNN


def load_model(path: str | Path, in_dim: int, edge_dim: int, hidden: int = 32) -> CascadeGNN:
    model = CascadeGNN(in_dim=in_dim, edge_dim=edge_dim, hidden=hidden)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_risk(model: CascadeGNN, data: Data) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)
    return dict(zip(data.node_order, probs.tolist()))
