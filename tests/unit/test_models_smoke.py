import torch

from cascaid.ingestion.schema import NODE_TYPE_ORDER, NUM_FEATURES
from cascaid.models.gnn import CascadeGNN

IN_DIM = NUM_FEATURES + len(NODE_TYPE_ORDER)


def _tiny_graph():
    x = torch.rand(3, IN_DIM)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, NUM_FEATURES)
    y = torch.tensor([0.0, 1.0, 0.0])
    return x, edge_index, edge_attr, y


def test_gnn_forward_pass_shape():
    x, edge_index, edge_attr, _ = _tiny_graph()
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=8, layers=2)
    out = model(x, edge_index, edge_attr)
    assert out.shape == (3,)


def test_gnn_one_training_step_runs():
    x, edge_index, edge_attr, y = _tiny_graph()
    model = CascadeGNN(in_dim=IN_DIM, edge_dim=NUM_FEATURES, hidden=8, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    opt.zero_grad()
    logits = model(x, edge_index, edge_attr)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
