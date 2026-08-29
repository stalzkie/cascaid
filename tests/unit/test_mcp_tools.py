import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch_geometric.data import Data

from cascaid.ingestion.graph_store import save_snapshot
from cascaid.mcp.tools import get_cascade_risk
from cascaid.storage.repository import init_db, record_scores


def _snapshot(run_id: str, step: int) -> Data:
    data = Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["agent", "store"]
    data.node_types = ["agent", "vector_store"]
    data.edges = [("agent", "store")]
    return data


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return sessionmaker(bind=engine)


def test_get_cascade_risk_returns_nodes_edges_and_current_risk(tmp_path):
    save_snapshot(_snapshot("run-1", step=2), tmp_path)
    session_factory = _session_factory()
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=2, scores={"agent": 0.2, "store": 0.9})

    result = get_cascade_risk(tmp_path, session_factory, run_id="run-1")

    assert result["run_id"] == "run-1"
    assert result["nodes"] == [
        {"name": "agent", "type": "agent", "risk_score": 0.2},
        {"name": "store", "type": "vector_store", "risk_score": 0.9},
    ]


def test_get_cascade_risk_returns_none_for_an_unknown_run(tmp_path):
    session_factory = _session_factory()

    assert get_cascade_risk(tmp_path, session_factory, run_id="no-such-run") is None
