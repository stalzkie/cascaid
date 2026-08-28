"""Read-only views composing the Graph Store (topology) and Storage (score history,
incidents) into what the frontend renders (PRD 5.2 Dashboard API / Frontend). No
model is loaded here -- current risk is whatever Model Serving last persisted."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from cascaid.ingestion.graph_store import latest_snapshot, list_runs
from cascaid.storage.repository import get_incidents, get_latest_scores, get_score_history


def list_runs_view(store_dir: str | Path) -> list[str]:
    return list_runs(store_dir)


def pipeline_view(store_dir: str | Path, session: Session, run_id: str) -> dict | None:
    data = latest_snapshot(store_dir, run_id)
    if data is None:
        return None
    scores = get_latest_scores(session, run_id)
    return {
        "run_id": run_id,
        "step": data.step,
        "nodes": [
            {"name": name, "type": node_type, "risk_score": scores.get(name)}
            for name, node_type in zip(data.node_order, data.node_types)
        ],
        "edges": [[caller, callee] for caller, callee in data.edges],
    }


def track_record_view(session: Session, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "history": [
            {
                "step": row.step,
                "node_name": row.node_name,
                "risk_score": row.risk_score,
                "predicted_at": row.predicted_at.isoformat(),
            }
            for row in get_score_history(session, run_id)
        ],
        "incidents": [
            {
                "node_name": row.node_name,
                "incident_type": row.incident_type,
                "occurred_at": row.occurred_at.isoformat(),
                "source": row.source,
            }
            for row in get_incidents(session, run_id)
        ],
    }
