"""Incident/degradation labeling (PRD Core Feature 2 / Phase 1 step 5).

Within a faulty run, the window between fault onset and full manifestation
("cascade_step") is the lead-time window this whole project is trying to
predict: risk is elevated but the failure hasn't fully landed yet. Steps at
or after cascade_step describe an already-manifested failure -- not the
prediction target -- so they're marked unusable rather than labeled.

Only the epicenter node and its direct upstream neighbors (nodes that
depend on it, i.e. its predecessors in the call graph) are labeled positive.
Every other node's own features never move during the fault, so correctly
labeling them requires reading a neighbor's state through graph structure --
this is what the GNN-vs-flattened-baseline and real-vs-shuffled-adjacency
comparisons in metrics.py are built to detect.
"""

from __future__ import annotations

import networkx as nx

EPICENTER = {
    "rate_limit_model": "primary_model",
    "vector_db_degradation": "vector_store",
}


def affected_nodes(scenario: str, static_graph: nx.DiGraph) -> set[str]:
    epicenter = EPICENTER.get(scenario)
    if epicenter is None:
        return set()
    return {epicenter} | set(static_graph.predecessors(epicenter))


def label_step(
    scenario: str,
    step: int,
    node_order: list[str],
    static_graph: nx.DiGraph,
    fault_onset_step: int | None,
    cascade_step: int | None,
) -> tuple[dict[str, int], dict[str, bool]]:
    labels = {n: 0 for n in node_order}
    usable = {n: True for n in node_order}

    if fault_onset_step is None:
        return labels, usable

    if step >= cascade_step:
        usable = {n: False for n in node_order}
        return labels, usable

    if step >= fault_onset_step:
        affected = affected_nodes(scenario, static_graph)
        for n in affected:
            labels[n] = 1

    return labels, usable
