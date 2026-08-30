"""Integration seam: import_langfuse_incidents against a real sqlite DB."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cascaid.ingestion.langfuse_import import import_langfuse_incidents
from cascaid.storage.repository import get_incidents, init_db


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return sessionmaker(bind=engine)


@pytest.mark.integration
def test_import_langfuse_incidents_records_only_degraded_scores():
    session_factory = _session_factory()
    scores = [
        {"name": "helpfulness", "value": 0.9, "dataType": "NUMERIC", "timestamp": "2026-08-01T00:00:00Z"},
        {"name": "toxicity", "value": 0.1, "dataType": "NUMERIC", "timestamp": "2026-08-02T03:00:00Z"},
    ]

    with session_factory() as session:
        count = import_langfuse_incidents(session, scores, run_id="run-1", node_name="agent-checkout", threshold=0.5)

    assert count == 1
    with session_factory() as session:
        incidents = get_incidents(session, run_id="run-1")
    assert len(incidents) == 1
    assert incidents[0].node_name == "agent-checkout"
    assert incidents[0].source == "langfuse"
    assert incidents[0].occurred_at.month == 8
    assert incidents[0].occurred_at.day == 2
    assert "toxicity" in incidents[0].incident_type
