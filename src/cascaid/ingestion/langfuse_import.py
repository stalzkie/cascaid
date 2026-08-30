"""Imports historical quality-degradation events from a Langfuse scores export
(PRD 4: historical incident/degradation labeling, "sourced from Langfuse/LangSmith/
Phoenix exports or manual import for MVP"). Reads the JSON shape Langfuse's own
public Scores API returns (id/name/value/dataType/timestamp) -- verified against
Langfuse's API documentation, not a UI "export" button whose exact file format
isn't independently documented anywhere.

A Langfuse score carries no concept of Cascaid's run_id/node_name -- a trace ID is
not a Cascaid pipeline-run identifier -- so every imported score is attributed to a
single (run_id, node_name) pair given explicitly by the caller. Auto-correlating
Langfuse traces to specific Cascaid pipeline nodes would need the customer's own
trace metadata to carry that mapping, which Cascaid has no way to assume exists.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from cascaid.storage.repository import record_incident


def parse_langfuse_scores(path: str | Path) -> list[dict]:
    """A bare JSON array of score objects, or {"data": [...]} -- both shapes
    appear across Langfuse's own API responses depending on endpoint/version."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    return raw


def scores_below_threshold(scores: list[dict], threshold: float) -> list[dict]:
    """Only NUMERIC scores below threshold count as a quality-degradation
    incident -- boolean/categorical/text scores aren't comparable to a numeric
    threshold, and a missing value can't be compared at all."""
    return [
        s
        for s in scores
        if s.get("dataType") == "NUMERIC" and isinstance(s.get("value"), int | float) and s["value"] < threshold
    ]


def import_langfuse_incidents(
    session: Session, scores: list[dict], run_id: str, node_name: str, threshold: float
) -> int:
    degraded = scores_below_threshold(scores, threshold)
    for score in degraded:
        record_incident(
            session,
            run_id=run_id,
            node_name=node_name,
            incident_type=f"langfuse_score_below_threshold:{score.get('name', 'unknown')}",
            occurred_at=datetime.fromisoformat(score["timestamp"]),
            source="langfuse",
        )
    return len(degraded)
