import torch
from torch_geometric.data import Data

from cascaid.ingestion.graph_store import latest_snapshot, list_runs, load_snapshot, save_snapshot


def _make_data(run_id: str = "run-1", step: int = 0) -> Data:
    data = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.5]], dtype=torch.float32),
    )
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["a", "b"]
    return data


def test_load_snapshot_returns_data_saved_by_save_snapshot(tmp_path):
    data = _make_data()

    path = save_snapshot(data, tmp_path)
    loaded = load_snapshot(path)

    assert torch.equal(loaded.x, data.x)
    assert torch.equal(loaded.edge_index, data.edge_index)
    assert torch.equal(loaded.edge_attr, data.edge_attr)
    assert loaded.run_id == data.run_id
    assert loaded.scenario == data.scenario
    assert loaded.step == data.step
    assert loaded.node_order == data.node_order


def test_latest_snapshot_returns_highest_step_for_run(tmp_path):
    for step in [0, 3, 1]:
        save_snapshot(_make_data(run_id="run-1", step=step), tmp_path)
    save_snapshot(_make_data(run_id="run-2", step=99), tmp_path)

    latest = latest_snapshot(tmp_path, "run-1")

    assert latest.step == 3
    assert latest.run_id == "run-1"


def test_latest_snapshot_returns_none_for_unknown_run(tmp_path):
    assert latest_snapshot(tmp_path, "no-such-run") is None


def test_list_runs_returns_sorted_run_ids_with_at_least_one_snapshot(tmp_path):
    save_snapshot(_make_data(run_id="run-b", step=0), tmp_path)
    save_snapshot(_make_data(run_id="run-a", step=0), tmp_path)

    assert list_runs(tmp_path) == ["run-a", "run-b"]


def test_list_runs_returns_empty_list_for_a_fresh_store_dir(tmp_path):
    assert list_runs(tmp_path) == []
