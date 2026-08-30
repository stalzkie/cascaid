"""Incident/degradation labeling (PRD Core Feature 2 / Phase 1 step 5).

Within a faulty run, the window between fault onset and full manifestation
("cascade_step") is the lead-time window this whole project is trying to
predict: risk is elevated but the failure hasn't fully landed yet. Steps at
or after cascade_step describe an already-manifested failure -- not the
prediction target -- so they're marked unusable rather than labeled.

Only a scenario's epicenter node(s) and their direct upstream neighbors
(nodes that depend on them, i.e. their predecessors in the call graph) are
labeled positive -- a scenario can have more than one simultaneous epicenter
(see EPICENTER["compound_cascade"]). Every other node's own features never
move during the fault, so correctly labeling them requires reading a
neighbor's state through graph structure -- this is what the
GNN-vs-flattened-baseline and real-vs-shuffled-adjacency comparisons in
metrics.py are built to detect.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx

EPICENTER: dict[str, tuple[str, ...]] = {
    "rate_limit_model": ("primary_model",),
    "vector_db_degradation": ("vector_store",),
    "cost_spike_model": ("primary_model",),
    "vector_store_flaky": ("vector_store",),
    "compound_cascade": ("primary_model", "vector_store"),
}

# Below this fraction of ramp progress, fault_progress() is close enough to
# zero that the affected edges/nodes are statistically indistinguishable
# from healthy -- see docs/GNN_Accuracy_Improvement_Log.md, Finding 3. Marked
# unusable rather than a hard positive, the same way the post-cascade window
# already is, instead of teaching the model a label it can't actually earn
# from the data.
RAMP_AMBIGUITY_CUTOFF = 0.5


def affected_nodes(scenario: str, static_graph: nx.DiGraph) -> set[str]:
    epicenters = EPICENTER.get(scenario, ())
    affected: set[str] = set()
    for epicenter in epicenters:
        affected |= {epicenter} | set(static_graph.predecessors(epicenter))
    return affected


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

    if step < fault_onset_step:
        return labels, usable

    ramp_steps = cascade_step - fault_onset_step
    progress = (step - fault_onset_step) / ramp_steps
    if progress < RAMP_AMBIGUITY_CUTOFF:
        usable = {n: False for n in node_order}
        return labels, usable

    affected = affected_nodes(scenario, static_graph)
    for n in affected:
        labels[n] = 1

    return labels, usable


def label_step_from_incidents(
    node_order: list[str],
    incidents: list[tuple[str, datetime]],
    step_start: datetime | None,
    step_end: datetime | None,
    window_before: timedelta = timedelta(minutes=5),
    window_after: timedelta = timedelta(minutes=5),
) -> tuple[dict[str, int], dict[str, bool]]:
    """Real-data counterpart to label_step() (see
    docs/Real_Data_Retraining_Plan.md) -- a real IncidentLabel has no
    fault_onset_step/cascade_step ramp, only a wall-clock occurred_at, so this
    marks a node positive when an incident for it falls within
    [step_start - window_before, step_end + window_after]. Deliberately
    node-local: unlike label_step's affected_nodes() propagation to
    predecessors, there's no verified basis for assuming a real incident
    cascades to callers the same way the synthetic scenarios were calibrated
    to. A step with no wall-clock bounds (occurred_at wasn't recorded for its
    events) can't be judged against any window, so it's marked unusable
    entirely rather than silently labeled 0."""
    labels = {n: 0 for n in node_order}
    usable = {n: True for n in node_order}

    if step_start is None or step_end is None:
        usable = {n: False for n in node_order}
        return labels, usable

    window_start = step_start - window_before
    window_end = step_end + window_after
    for node_name, occurred_at in incidents:
        if node_name in labels and window_start <= occurred_at <= window_end:
            labels[node_name] = 1

    return labels, usable
