import pytest
import torch
from torch_geometric.data import Data

from cascaid.models.gnn import CascadeGNN
from cascaid.models.model_config import ModelConfig, save_model_config
from cascaid.serving.risk import load_model, predict_risk

IN_DIM = 6
EDGE_DIM = 4


def _data() -> Data:
    data = Data(
        x=torch.rand(3, IN_DIM),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.rand(2, EDGE_DIM),
    )
    data.node_order = ["a", "b", "c"]
    return data


def test_predict_risk_returns_one_score_per_node_in_zero_one_range():
    torch.manual_seed(0)
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)

    scores = predict_risk(model, _data())

    assert set(scores.keys()) == {"a", "b", "c"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_load_model_reproduces_saved_model_scores(tmp_path):
    torch.manual_seed(1)
    trained = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    ckpt_path = tmp_path / "model.pt"
    torch.save(trained.state_dict(), ckpt_path)

    loaded = load_model(ckpt_path, in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=8)
    data = _data()

    assert predict_risk(loaded, data) == predict_risk(trained, data)


def test_load_model_defaults_to_hidden_32_layers_2_conv_gine_when_no_sidecar_and_no_override(tmp_path):
    # Unchanged legacy behavior: a .pt saved before model_config.py existed (or by
    # anything that doesn't call save_model_config) has no sidecar, so load_model must
    # keep working exactly as it did before this feature -- no forced migration.
    torch.manual_seed(2)
    trained = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM)  # hidden=32, layers=2, conv="gine" defaults
    ckpt_path = tmp_path / "model.pt"
    torch.save(trained.state_dict(), ckpt_path)

    loaded = load_model(ckpt_path, in_dim=IN_DIM, edge_dim=EDGE_DIM)
    data = _data()

    assert predict_risk(loaded, data) == predict_risk(trained, data)


def test_load_model_auto_configures_from_sidecar_when_no_override_given(tmp_path):
    # The actual fix: a model trained with non-default hyperparameters (e.g. via
    # scripts/gnn_experiment.py's sweeps) can be loaded with zero flags -- the sidecar
    # supplies hidden/layers/conv instead of the caller needing to already know them.
    torch.manual_seed(3)
    trained = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16, layers=3, conv="gat")
    ckpt_path = tmp_path / "model.pt"
    torch.save(trained.state_dict(), ckpt_path)
    save_model_config(ModelConfig(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16, layers=3, conv="gat"), ckpt_path)

    loaded = load_model(ckpt_path, in_dim=IN_DIM, edge_dim=EDGE_DIM)
    data = _data()

    assert predict_risk(loaded, data) == predict_risk(trained, data)


def test_load_model_raises_a_clear_error_when_an_explicit_override_conflicts_with_the_sidecar(tmp_path):
    torch.manual_seed(4)
    trained = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16)
    ckpt_path = tmp_path / "model.pt"
    torch.save(trained.state_dict(), ckpt_path)
    save_model_config(ModelConfig(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16, layers=2, conv="gine"), ckpt_path)

    with pytest.raises(ValueError, match="hidden=16.*hidden=32"):
        load_model(ckpt_path, in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=32)


def test_load_model_explicit_override_matching_the_sidecar_is_not_an_error(tmp_path):
    torch.manual_seed(5)
    trained = CascadeGNN(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16)
    ckpt_path = tmp_path / "model.pt"
    torch.save(trained.state_dict(), ckpt_path)
    save_model_config(ModelConfig(in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16, layers=2, conv="gine"), ckpt_path)

    loaded = load_model(ckpt_path, in_dim=IN_DIM, edge_dim=EDGE_DIM, hidden=16)
    data = _data()

    assert predict_risk(loaded, data) == predict_risk(trained, data)
