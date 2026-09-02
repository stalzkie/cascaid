"""GNN inference over a graph snapshot -> per-node cascade risk score (PRD 5.2
Model Serving). Kept free of FastAPI so it can be unit-tested without HTTP."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from cascaid.models.gnn import CascadeGNN
from cascaid.models.model_config import load_model_config


def load_model(
    path: str | Path,
    in_dim: int,
    edge_dim: int,
    hidden: int | None = None,
    layers: int | None = None,
    conv: str | None = None,
) -> CascadeGNN:
    """hidden/layers/conv default to None ("not specified"), not CascadeGNN's own
    defaults directly: this lets a sidecar .config.json (see model_config.py) saved by
    train.py/retrain.py supply the model's actual hyperparameters automatically, so a
    model trained with a non-default hidden/layers/conv (e.g. via
    scripts/gnn_experiment.py's sweeps) loads correctly with zero flags. An explicit
    caller-provided value that conflicts with the sidecar raises a clear error instead
    of PyTorch's raw "size mismatch" deep inside load_state_dict -- see
    docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md. No sidecar and
    no override falls back to CascadeGNN's own defaults, unchanged from before this
    existed (a .pt saved before model_config.py existed still loads)."""
    config = load_model_config(path)

    if config is not None:
        _check_no_conflict("hidden", hidden, config.hidden)
        _check_no_conflict("layers", layers, config.layers)
        _check_no_conflict("conv", conv, config.conv)

    resolved_hidden = hidden if hidden is not None else (config.hidden if config else 32)
    resolved_layers = layers if layers is not None else (config.layers if config else 2)
    resolved_conv = conv if conv is not None else (config.conv if config else "gine")

    model = CascadeGNN(
        in_dim=in_dim, edge_dim=edge_dim, hidden=resolved_hidden, layers=resolved_layers, conv=resolved_conv
    )
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def _check_no_conflict(name: str, requested, actual) -> None:
    if requested is not None and requested != actual:
        raise ValueError(
            f"Model was trained with {name}={actual!r}, but was asked to load with {name}={requested!r}. "
            f"Omit --{name} to use the model's own recorded config, or pass a matching value."
        )


def predict_risk(model: CascadeGNN, data: Data) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.sigmoid(logits)
    return dict(zip(data.node_order, probs.tolist()))
