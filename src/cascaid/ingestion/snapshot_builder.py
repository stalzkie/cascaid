"""Rolling-window feature aggregation: turns one run's raw CallEvents into a
time-indexed series of graph snapshots (PRD 5.2 Graph Store)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from cascaid.ingestion.schema import NODE_TYPE_ORDER, NOMINAL_DEFAULTS, NUM_FEATURES, CallEvent, NodeType

WINDOW = 5


def _rolling_stats(history: deque) -> tuple[float, float, float, float]:
    if not history:
        return None  # type: ignore[return-value]
    latencies = [h[0] for h in history]
    errors = [h[1] for h in history]
    retries = [h[2] for h in history]
    costs = [h[3] for h in history]
    return (
        float(np.mean(latencies)),
        float(np.mean(errors)),
        float(np.mean(retries)),
        float(np.mean(costs)),
    )


def _default_for(edge: tuple[str, str], edges_kind: dict[tuple[str, str], str]) -> tuple[float, float, float, float]:
    kind = edges_kind[edge]
    return NOMINAL_DEFAULTS[kind]


def _edge_kind(caller_type: NodeType, callee_type: NodeType) -> str:
    if callee_type == NodeType.VECTOR_STORE:
        return "vector_store"
    if callee_type == NodeType.MODEL_ENDPOINT:
        return "model_endpoint"
    return "control"


@dataclass
class Snapshot:
    run_id: str
    scenario: str
    step: int
    node_order: list[str]
    node_features: np.ndarray  # [num_nodes, NUM_FEATURES]
    node_type_onehot: np.ndarray  # [num_nodes, len(NODE_TYPE_ORDER)]
    edge_index: np.ndarray  # [2, num_edges]
    edge_features: np.ndarray  # [num_edges, NUM_FEATURES]


def build_snapshots(
    nodes: dict[str, NodeType],
    edges: list[tuple[str, str]],
    events: list[CallEvent],
    window: int = WINDOW,
) -> list[Snapshot]:
    node_order = list(nodes.keys())
    node_index = {n: i for i, n in enumerate(node_order)}
    edges_kind = {(c, cal): _edge_kind(nodes[c], nodes[cal]) for c, cal in edges}

    events_by_step_edge: dict[tuple[int, tuple[str, str]], list[CallEvent]] = defaultdict(list)
    max_step = -1
    for ev in events:
        events_by_step_edge[(ev.step, (ev.caller, ev.callee))].append(ev)
        max_step = max(max_step, ev.step)

    history: dict[tuple[str, str], deque] = {e: deque(maxlen=window) for e in edges}
    type_onehot = np.zeros((len(node_order), len(NODE_TYPE_ORDER)), dtype=np.float32)
    for name, i in node_index.items():
        type_onehot[i, NODE_TYPE_ORDER.index(nodes[name])] = 1.0

    edge_list = list(edges)
    edge_index_np = np.array(
        [[node_index[c] for c, _ in edge_list], [node_index[cal] for _, cal in edge_list]],
        dtype=np.int64,
    )

    incoming: dict[str, list[int]] = defaultdict(list)
    for ei, (c, cal) in enumerate(edge_list):
        incoming[cal].append(ei)

    snapshots = []
    for step in range(max_step + 1):
        edge_feats = np.zeros((len(edge_list), NUM_FEATURES), dtype=np.float32)
        for ei, edge in enumerate(edge_list):
            new_events = events_by_step_edge.get((step, edge), [])
            if new_events:
                lat = float(np.mean([e.latency_ms for e in new_events]))
                err = float(np.mean([1.0 if e.error else 0.0 for e in new_events]))
                retr = float(np.mean([1.0 if e.retried else 0.0 for e in new_events]))
                cost = float(np.mean([e.token_cost for e in new_events]))
                history[edge].append((lat, err, retr, cost))
            stats = _rolling_stats(history[edge])
            if stats is None:
                stats = _default_for(edge, edges_kind)
            edge_feats[ei] = stats

        node_feats = np.zeros((len(node_order), NUM_FEATURES), dtype=np.float32)
        for name, i in node_index.items():
            in_edges = incoming[name]
            if in_edges:
                node_feats[i] = edge_feats[in_edges].mean(axis=0)
            else:
                node_feats[i] = NOMINAL_DEFAULTS["control"]

        snapshots.append(
            Snapshot(
                run_id=events[0].run_id if events else "unknown",
                scenario=events[0].scenario if events else "unknown",
                step=step,
                node_order=node_order,
                node_features=node_feats,
                node_type_onehot=type_onehot,
                edge_index=edge_index_np,
                edge_features=edge_feats,
            )
        )
    return snapshots


def shuffle_edge_index(edge_index: np.ndarray, num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of targets, preserving edge count -- used for the
    real-vs-shuffled-adjacency ablation (PRD 6.2)."""
    shuffled = edge_index.copy()
    num_edges = edge_index.shape[1]
    if num_edges <= num_nodes:
        shuffled[1] = rng.permutation(num_nodes)[:num_edges]
    else:
        shuffled[1] = rng.integers(0, num_nodes, size=num_edges)
    return shuffled


def to_pyg_data(
    snapshot: Snapshot,
    labels: dict[str, int] | None = None,
    usable: dict[str, bool] | None = None,
    edge_index_override: np.ndarray | None = None,
) -> Data:
    x = np.concatenate([snapshot.node_features, snapshot.node_type_onehot], axis=1)
    base_edge_index = edge_index_override if edge_index_override is not None else snapshot.edge_index
    # PyG message passing propagates source -> target (edge_index[0] -> edge_index[1]).
    # Our edges are caller -> callee, but cascade risk propagates the other way (a
    # degrading callee is what its caller needs to learn about), so include the
    # reverse direction too -- otherwise the GNN never actually sees the signal it's
    # supposed to detect and performs no better than random adjacency.
    edge_index = np.concatenate([base_edge_index, base_edge_index[[1, 0]]], axis=1)
    edge_attr = np.concatenate([snapshot.edge_features, snapshot.edge_features], axis=0)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    if labels is not None:
        data.y = torch.tensor([labels[n] for n in snapshot.node_order], dtype=torch.float32)
    if usable is not None:
        data.usable = torch.tensor([usable[n] for n in snapshot.node_order], dtype=torch.bool)
    data.run_id = snapshot.run_id
    data.scenario = snapshot.scenario
    data.step = snapshot.step
    data.node_order = snapshot.node_order
    return data
