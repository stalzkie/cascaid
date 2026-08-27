"""GATConv/GINEConv-based cascade risk GNN (PRD 5.2 Model Serving).

GINEConv is used because it natively consumes edge_attr, which is how the
PRD's edge features (latency, error rate, token cost, retry rate) enter the
model -- not just node features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv


class CascadeGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 32, layers: int = 2, conv: str = "gine"):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_proj = nn.ModuleList([nn.Linear(edge_dim, hidden) for _ in range(layers)])
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if conv == "gat":
                self.convs.append(GATConv(hidden, hidden, edge_dim=hidden))
            else:
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINEConv(mlp))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for conv, eproj in zip(self.convs, self.edge_proj):
            h = F.relu(conv(h, edge_index, eproj(edge_attr)))
        return self.out(h).squeeze(-1)
