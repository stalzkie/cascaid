from datetime import datetime, timezone

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torch_geometric.data import Data

from cascaid.dashboard.views import pipeline_view, track_record_view
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.storage.repository import init_db, record_incident, record_scores


def _snapshot(run_id: str, step: int) -> Data:
    data = Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.zeros(1, 1))
    data.run_id = run_id
    data.scenario = "baseline"
    data.step = step
    data.node_order = ["agent", "store"]
    data.node_types = ["agent", "vector_store"]
    data.edges = [("agent", "store")]
    return data


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def test_pipeline_view_returns_none_when_no_snapshot_persisted(tmp_path):
    session = _session()

    assert pipeline_view(tmp_path, session, run_id="no-such-run") is None


def test_pipeline_view_returns_nodes_edges_and_current_risk(tmp_path):
    save_snapshot(_snapshot("run-1", step=2), tmp_path)
    session = _session()
    record_scores(session, run_id="run-1", step=2, scores={"agent": 0.2, "store": 0.9})

    view = pipeline_view(tmp_path, session, run_id="run-1")

    assert view["run_id"] == "run-1"
    assert view["step"] == 2
    assert view["nodes"] == [
        {"name": "agent", "type": "agent", "risk_score": 0.2},
        {"name": "store", "type": "vector_store", "risk_score": 0.9},
    ]
    assert view["edges"] == [["agent", "store"]]


def test_pipeline_view_risk_score_is_none_when_not_yet_scored(tmp_path):
    save_snapshot(_snapshot("run-1", step=0), tmp_path)
    session = _session()

    view = pipeline_view(tmp_path, session, run_id="run-1")

    assert view["nodes"][0]["risk_score"] is None


def test_track_record_view_returns_score_history_and_incidents():
    session = _session()
    record_scores(session, run_id="run-1", step=0, scores={"agent": 0.1})
    record_incident(
        session,
        run_id="run-1",
        node_name="agent",
        incident_type="degradation",
        occurred_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        source="manual",
    )

    view = track_record_view(session, run_id="run-1")

    assert view["history"] == [{"step": 0, "node_name": "agent", "risk_score": 0.1}]
    assert len(view["incidents"]) == 1
    assert view["incidents"][0]["node_name"] == "agent"
