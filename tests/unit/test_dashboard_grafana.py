import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch_geometric.data import Data

from cascaid.dashboard.grafana import grafana_query, grafana_search
from cascaid.ingestion.graph_store import save_snapshot
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


def test_grafana_search_returns_run_node_target_strings_for_every_known_run(tmp_path):
    save_snapshot(_snapshot("run-a", step=0), tmp_path)
    save_snapshot(_snapshot("run-b", step=0), tmp_path)

    targets = grafana_search(tmp_path)

    assert targets == ["run-a/agent", "run-a/store", "run-b/agent", "run-b/store"]


def test_grafana_query_returns_datapoints_for_a_known_target():
    session_factory = _session_factory()
    with session_factory() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.2})
        record_scores(session, run_id="run-1", step=1, scores={"agent": 0.5})

    result = grafana_query(session_factory, targets=["run-1/agent"])

    assert len(result) == 1
    assert result[0]["target"] == "run-1/agent"
    datapoints = result[0]["datapoints"]
    assert [value for value, _ in datapoints] == [0.2, 0.5]
    timestamps = [ts for _, ts in datapoints]
    assert timestamps == sorted(timestamps)
    assert all(isinstance(ts, int) for ts in timestamps)


def test_grafana_query_rejects_a_target_without_a_run_id():
    session_factory = _session_factory()

    with pytest.raises(ValueError):
        grafana_query(session_factory, targets=["agent"])


def test_grafana_query_returns_empty_datapoints_for_an_unknown_run_or_node():
    session_factory = _session_factory()

    result = grafana_query(session_factory, targets=["no-such-run/ghost"])

    assert result == [{"target": "no-such-run/ghost", "datapoints": []}]
