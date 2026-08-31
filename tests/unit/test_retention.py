from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from cascaid.storage.models import AlertHistory, IncidentLabel, ScoreHistory
from cascaid.storage.repository import init_db
from cascaid.storage.retention import DEFAULT_RETENTION_DAYS, delete_expired_history


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


def test_delete_expired_history_removes_rows_older_than_the_retention_window():
    session = _session()
    session.add(ScoreHistory(run_id="old", step=0, node_name="agent", risk_score=0.1, predicted_at=_days_ago(100)))
    session.add(ScoreHistory(run_id="new", step=0, node_name="agent", risk_score=0.2, predicted_at=_days_ago(1)))
    session.commit()

    deleted = delete_expired_history(session, retention_days=90)

    assert deleted["score_history"] == 1
    remaining = list(session.scalars(select(ScoreHistory)))
    assert [row.run_id for row in remaining] == ["new"]


def test_delete_expired_history_keeps_rows_newer_than_the_retention_window():
    session = _session()
    session.add(ScoreHistory(run_id="new", step=0, node_name="agent", risk_score=0.2, predicted_at=_days_ago(1)))
    session.commit()

    deleted = delete_expired_history(session, retention_days=90)

    assert deleted["score_history"] == 0
    assert len(list(session.scalars(select(ScoreHistory)))) == 1


def test_delete_expired_history_covers_incidents_and_alerts_too():
    session = _session()
    session.add(
        IncidentLabel(
            run_id="old", node_name="agent", incident_type="degradation", occurred_at=_days_ago(200), source="manual"
        )
    )
    session.add(
        AlertHistory(
            run_id="old",
            node_name="agent",
            risk_score=0.9,
            message="at risk",
            channel="webhook",
            sent_at=_days_ago(200),
        )
    )
    session.commit()

    deleted = delete_expired_history(session, retention_days=90)

    assert deleted["incident_labels"] == 1
    assert deleted["alert_history"] == 1
    assert list(session.scalars(select(IncidentLabel))) == []
    assert list(session.scalars(select(AlertHistory))) == []


def test_default_retention_days_is_conservative():
    # A regression guard against silently shrinking the default -- ADR 0004 explicitly
    # flags picking too short a default as a real risk (deleting history a customer
    # still wanted for drift comparison), not just a storage-efficiency knob.
    assert DEFAULT_RETENTION_DAYS >= 90
