"""Integration seam: the real MCPServer instance cascaid.mcp.server builds from
argv, exercised through the SDK's own call_tool dispatch (PRD 7 MCP exposure)."""

import asyncio

import pytest
import torch
from torch_geometric.data import Data

import cascaid.mcp.server as mcp_server_cli
from cascaid.ingestion.graph_store import save_snapshot
from cascaid.storage.db import get_engine, make_session_factory
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


@pytest.mark.integration
def test_cascade_risk_tool_returns_current_scores_for_a_known_run(tmp_path):
    store_dir = tmp_path / "graph_store"
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))
    save_snapshot(_snapshot("run-1", step=0), store_dir)
    with make_session_factory(database_url)() as session:
        record_scores(session, run_id="run-1", step=0, scores={"agent": 0.2, "store": 0.9})

    server = mcp_server_cli.build_server_from_argv(["--store", str(store_dir), "--database-url", database_url])

    result = asyncio.run(server.call_tool("cascade_risk", {"run_id": "run-1"}))

    assert result.is_error is False
    assert result.structured_content["result"]["nodes"] == [
        {"name": "agent", "type": "agent", "risk_score": 0.2},
        {"name": "store", "type": "vector_store", "risk_score": 0.9},
    ]


@pytest.mark.integration
def test_cascade_risk_tool_returns_none_for_an_unknown_run(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    server = mcp_server_cli.build_server_from_argv(
        ["--store", str(tmp_path / "graph_store"), "--database-url", database_url]
    )

    result = asyncio.run(server.call_tool("cascade_risk", {"run_id": "no-such-run"}))

    assert result.structured_content["result"] is None
