"""Integration seam: the real LangGraph demo pipeline -> topology-graph snapshots
-> labeling -> PyG Data, wired together end-to-end (not hand-built fixtures)."""

import numpy as np
import pytest

from cascaid.ingestion.labeling import label_step
from cascaid.ingestion.snapshot_builder import build_snapshots, to_pyg_data
from cascaid.ingestion.topology import build_static_graph
from cascaid_demo.fault_injection import make_scenario
from cascaid_demo.mock_llm_gateway import ModelGateway
from cascaid_demo.mock_vector_db import VectorStore
from cascaid_demo.pipeline import ALL_EDGES, STATIC_NODES, build_pipeline
from cascaid_demo.recorder import Recorder

TOTAL_STEPS = 30


@pytest.mark.integration
def test_rate_limit_scenario_end_to_end_produces_positive_and_negative_labels():
    rng = np.random.default_rng(7)
    scenario = make_scenario("rate_limit_model", TOTAL_STEPS, rng, ramp_steps=8)
    recorder = Recorder()
    graph = build_pipeline()
    vector_store = VectorStore()
    gateway = ModelGateway()
    state = {"query": "", "retrieved_context": "", "research_notes": "", "answer": ""}

    for step in range(TOTAL_STEPS):
        config = {
            "configurable": {
                "recorder": recorder,
                "scenario": scenario,
                "step": step,
                "rng": rng,
                "vector_store": vector_store,
                "gateway": gateway,
                "run_id": "it-run-1",
            }
        }
        state = graph.invoke(state, config=config)

    assert len(recorder.events) > 0

    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    assert len(snapshots) == TOTAL_STEPS

    static_graph = build_static_graph(STATIC_NODES, ALL_EDGES)
    cascade_step = scenario.fault_onset_step + scenario.ramp_steps

    seen_positive, seen_negative = False, False
    for snap in snapshots:
        labels, usable = label_step(
            "rate_limit_model",
            snap.step,
            snap.node_order,
            static_graph,
            scenario.fault_onset_step,
            cascade_step,
        )
        data = to_pyg_data(snap, labels=labels, usable=usable)

        assert data.x.shape[0] == len(STATIC_NODES)
        assert data.edge_index.shape[1] == 2 * len(ALL_EDGES)  # bidirectional
        assert data.y.shape[0] == len(STATIC_NODES)

        if usable["primary_model"] and labels["primary_model"] == 1:
            seen_positive = True
        if labels["primary_model"] == 0:
            seen_negative = True

    assert seen_positive, "ramp window should have produced at least one positive label"
    assert seen_negative, "steps before fault onset should have produced negative labels"


@pytest.mark.integration
def test_baseline_scenario_never_produces_positive_labels():
    rng = np.random.default_rng(11)
    scenario = make_scenario("baseline", TOTAL_STEPS, rng)
    recorder = Recorder()
    graph = build_pipeline()
    vector_store = VectorStore()
    gateway = ModelGateway()
    state = {"query": "", "retrieved_context": "", "research_notes": "", "answer": ""}

    for step in range(TOTAL_STEPS):
        config = {
            "configurable": {
                "recorder": recorder,
                "scenario": scenario,
                "step": step,
                "rng": rng,
                "vector_store": vector_store,
                "gateway": gateway,
                "run_id": "it-run-2",
            }
        }
        state = graph.invoke(state, config=config)

    snapshots = build_snapshots(STATIC_NODES, ALL_EDGES, recorder.events)
    static_graph = build_static_graph(STATIC_NODES, ALL_EDGES)

    for snap in snapshots:
        labels, usable = label_step(
            "baseline",
            snap.step,
            snap.node_order,
            static_graph,
            None,
            None,
        )
        assert all(v == 0 for v in labels.values())
        assert all(usable.values())
