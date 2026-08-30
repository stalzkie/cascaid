import numpy as np
import torch
from torch_geometric.data import Data

from cascaid.drift import check_drift
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.ingestion.schema import FEATURE_NAMES, NUM_FEATURES
from cascaid.serving.drift import compute_reference


def _snapshot(run_id: str, step: int, feature_value: float) -> Data:
    x = np.full((2, NUM_FEATURES + 4), feature_value, dtype=np.float32)
    data = Data(x=torch.tensor(x), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["a", "b"]
    data.node_types = ["agent", "tool"]
    data.edges = [("a", "b")]
    return data


def test_check_drift_reads_all_snapshots_for_a_run_and_scores_against_the_reference(tmp_path):
    rng = np.random.default_rng(0)
    reference_features = rng.uniform(0, 1, size=(500, NUM_FEATURES))
    reference = compute_reference(reference_features, feature_names=FEATURE_NAMES)

    save_snapshot(_snapshot("run-1", step=0, feature_value=0.5), tmp_path)
    save_snapshot(_snapshot("run-1", step=1, feature_value=0.5), tmp_path)

    scores = check_drift(tmp_path, "run-1", reference)

    assert set(scores.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in scores.values())


def test_check_drift_returns_empty_when_no_snapshots_exist_for_the_run(tmp_path):
    rng = np.random.default_rng(0)
    reference = compute_reference(rng.uniform(0, 1, size=(100, NUM_FEATURES)), feature_names=FEATURE_NAMES)

    scores = check_drift(tmp_path, "no-such-run", reference)

    assert scores == {}
