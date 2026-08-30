"""E2E seam: the real `cascaid drift` CLI entry point, argv to stdout, against a
real Graph Store snapshot and a real reference file on disk."""

import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

import cascaid.drift as drift_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import FEATURE_NAMES, NUM_FEATURES
from cascaid.serving.drift import compute_reference, save_reference


def _snapshot(run_id: str, step: int) -> Data:
    x = np.full((2, NUM_FEATURES + 4), 0.5, dtype=np.float32)
    data = Data(x=torch.tensor(x), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["a", "b"]
    data.node_types = ["agent", "tool"]
    data.edges = [("a", "b")]
    return data


@pytest.mark.e2e
def test_drift_cli_reports_scores_for_a_real_run(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "graph_store"
    save_snapshot(_snapshot("run-1", step=0), store_dir)

    rng = np.random.default_rng(0)
    reference = compute_reference(rng.uniform(0, 1, size=(500, NUM_FEATURES)), feature_names=FEATURE_NAMES)
    reference_path = tmp_path / "reference.json"
    save_reference(reference, reference_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["drift", "--store", str(store_dir), "--reference", str(reference_path), "--run-id", "run-1"],
    )

    drift_cli.main()

    output = capsys.readouterr().out
    assert "Drift report for run_id=run-1" in output
    for name in FEATURE_NAMES:
        assert name in output


@pytest.mark.e2e
def test_drift_cli_reports_no_snapshots_for_an_unknown_run(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "graph_store"
    rng = np.random.default_rng(0)
    reference = compute_reference(rng.uniform(0, 1, size=(100, NUM_FEATURES)), feature_names=FEATURE_NAMES)
    reference_path = tmp_path / "reference.json"
    save_reference(reference, reference_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["drift", "--store", str(store_dir), "--reference", str(reference_path), "--run-id", "no-such-run"],
    )

    drift_cli.main()

    output = capsys.readouterr().out
    assert "No snapshots found" in output
