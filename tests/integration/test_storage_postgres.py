"""Integration seam: the storage schema against a real Postgres, not just SQLite --
dialect differences (timezone-aware timestamps, autoincrement) matter for a schema
that only ever runs against SQLite in the unit suite. Skipped unless
CASCAID_TEST_DATABASE_URL is set (CI provides a postgres service container for the
integration job; see .github/workflows/ci.yml)."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from cascaid.storage.db import get_engine
from cascaid.storage.repository import (
    get_alert_history,
    get_incidents,
    get_score_history,
    init_db,
    record_alert,
    record_incident,
    record_scores,
)

DATABASE_URL = os.environ.get("CASCAID_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="CASCAID_TEST_DATABASE_URL not set")


@pytest.mark.integration
def test_score_history_round_trips_through_real_postgres():
    engine = get_engine(DATABASE_URL)
    init_db(engine)
    with Session(engine) as session:
        record_scores(session, run_id="pg-run", step=1, scores={"agent": 0.42})

        history = get_score_history(session, run_id="pg-run")

        assert [(row.node_name, row.risk_score) for row in history] == [("agent", 0.42)]


@pytest.mark.integration
def test_incident_and_alert_history_round_trip_through_real_postgres():
    engine = get_engine(DATABASE_URL)
    init_db(engine)
    with Session(engine) as session:
        record_incident(
            session,
            run_id="pg-run-2",
            node_name="model",
            incident_type="degradation",
            occurred_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
            source="manual",
        )
        record_alert(
            session, run_id="pg-run-2", node_name="model", risk_score=0.9, message="at risk", channel="webhook"
        )

        incidents = get_incidents(session, run_id="pg-run-2")
        alerts = get_alert_history(session, run_id="pg-run-2")

        assert [i.incident_type for i in incidents] == ["degradation"]
        assert [a.channel for a in alerts] == ["webhook"]
