"""Sidecar config recording the hyperparameters a saved GNN's weights were trained
with (hidden/layers/conv, plus in_dim/edge_dim for completeness) -- a `state_dict`
alone only records values, not the shape those values assume, so loading it with the
wrong hidden/layers/conv fails deep inside PyTorch with a raw tensor-shape mismatch
instead of a message naming the actual problem (see
docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md).

Saved as JSON next to the .pt weights (out_path.with_suffix(".config.json")) -- the
same sibling-file convention train.py already uses for the drift reference
(.drift_reference.json), not a change to the weights file's own format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int
    edge_dim: int
    hidden: int
    layers: int
    conv: str


def config_path_for(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".config.json")


def save_model_config(config: ModelConfig, model_path: str | Path) -> None:
    config_path_for(model_path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_model_config(model_path: str | Path) -> ModelConfig | None:
    path = config_path_for(model_path)
    if not path.exists():
        return None
    return ModelConfig(**json.loads(path.read_text(encoding="utf-8")))
