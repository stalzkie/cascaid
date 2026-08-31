"""Integration seam: the storage schema against a real Postgres, not just SQLite --
dialect differences (timezone-aware timestamps, autoincrement) matter for a schema
that only ever runs against SQLite in the unit suite. Skipped unless
CASCAID_TEST_DATABASE_URL is set (CI provides a postgres service container for the
integration job; see .github/workflows/ci.yml)."""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
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


def _reflect_schema(engine) -> dict[str, dict[str, str]]:
    inspector = inspect(engine)
    return {
        table: {col["name"]: str(col["type"]) for col in inspector.get_columns(table)}
        for table in sorted(inspector.get_table_names())
        if table not in ("alembic_version",)
    }


@pytest.mark.integration
def test_alembic_baseline_matches_create_all_schema_on_real_postgres():
    """Same guarantee as tests/unit/test_alembic_migrations.py, but against real
    Postgres -- dialect-specific behavior (e.g. Postgres's own autoincrement/serial
    handling) only shows up here, not against SQLite."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    create_all_engine = get_engine(DATABASE_URL)
    init_db(create_all_engine)
    create_all_schema = _reflect_schema(create_all_engine)

    alembic_url = DATABASE_URL.rsplit("/", 1)[0] + "/cascaid_alembic_parity_test"
    admin_engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")
    db_name = alembic_url.rsplit("/", 1)[1]
    with admin_engine.connect() as conn:
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{db_name}"')
        conn.exec_driver_sql(f'CREATE DATABASE "{db_name}"')
    try:
        env = {**os.environ, "DATABASE_URL": alembic_url}
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
        )
        alembic_schema = _reflect_schema(create_engine(alembic_url))
    finally:
        with admin_engine.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{db_name}"')

    assert alembic_schema == create_all_schema


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
