import torch
from torch_geometric.data import Data

from cascaid.models.gnn import CascadeGNN
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
