"""Grafana JSON-datasource adapter (PRD 4.7/7): lets a Grafana panel query Cascaid's
risk signal via the SimPod-json-datasource/Infinity plugin convention, so no custom
Grafana panel plugin has to be built, shipped, or signed. Same shape as mcp/tools.py:
a thin wrapper reusing the existing graph_store/repository seams, no new business logic.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from cascaid.ingestion.graph_store import latest_snapshot, list_runs
from cascaid.storage.repository import get_score_history


def grafana_search(store_dir: str | Path) -> list[str]:
    """'run_id/node_name' target strings for every node in every known run's latest
    snapshot -- what Grafana's query editor offers to autocomplete."""
    targets: list[str] = []
    for run_id in list_runs(store_dir):
        data = latest_snapshot(store_dir, run_id)
        if data is None:
            continue
        targets.extend(f"{run_id}/{name}" for name in data.node_order)
    return targets


def grafana_query(session_factory: sessionmaker[Session], targets: list[str]) -> list[dict]:
    """Grafana timeserie response for each 'run_id/node_name' target: the node's full
    risk-score history as [value, epoch_ms] datapoints, oldest first."""
    series = []
    with session_factory() as session:
        for target in targets:
            run_id, sep, node_name = target.partition("/")
            if not sep:
                raise ValueError(f"invalid target {target!r}, expected 'run_id/node_name'")
            history = get_score_history(session, run_id, node_name)
            datapoints = [[row.risk_score, int(row.predicted_at.timestamp() * 1000)] for row in history]
            series.append({"target": target, "datapoints": datapoints})
    return series
