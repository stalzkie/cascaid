"""E2E seam: the real CLI entry points a user runs, wired end-to-end with tiny
params -- no source changes needed since both main()s already parse sys.argv."""

import sys

import pytest
import torch

import cascaid.train as train_cli
import cascaid_demo.run_scenarios as run_scenarios_cli


@pytest.mark.e2e
def test_demo_and_train_cli_run_end_to_end(tmp_path, monkeypatch):
    data_dir = tmp_path / "runs"
    model_path = tmp_path / "models" / "pretrained_base.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scenarios",
            "--runs-per-scenario",
            "3",
            "--steps",
            "20",
            "--out",
            str(data_dir),
            "--seed",
            "0",
        ],
    )
    run_scenarios_cli.main()

    assert (data_dir / "topology.json").exists()
    assert (data_dir / "manifest.jsonl").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(data_dir),
            "--epochs",
            "3",
            "--out",
            str(model_path),
        ],
    )
    train_cli.main()

    assert model_path.exists()
    state_dict = torch.load(model_path, weights_only=True)
    assert len(state_dict) > 0


@pytest.mark.e2e
def test_train_cli_is_reproducible_given_the_same_seed(tmp_path, monkeypatch):
    data_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_scenarios", "--runs-per-scenario", "3", "--steps", "20", "--out", str(data_dir), "--seed", "0"],
    )
    run_scenarios_cli.main()

    model_path_a = tmp_path / "a.pt"
    model_path_b = tmp_path / "b.pt"
    for model_path in (model_path_a, model_path_b):
        monkeypatch.setattr(
            sys,
            "argv",
            ["train", "--data", str(data_dir), "--epochs", "3", "--seed", "42", "--out", str(model_path)],
        )
        train_cli.main()

    state_dict_a = torch.load(model_path_a, weights_only=True)
    state_dict_b = torch.load(model_path_b, weights_only=True)
    assert state_dict_a.keys() == state_dict_b.keys()
    for key in state_dict_a:
        assert torch.equal(state_dict_a[key], state_dict_b[key]), f"{key} differs between runs with the same seed"
