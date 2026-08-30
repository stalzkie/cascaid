from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from cascaid.storage.repository import (
    create_session,
    delete_session,
    get_alert_history,
    get_config,
    get_incidents,
    get_latest_scores,
    get_score_history,
    get_session,
    init_db,
    record_alert,
    record_incident,
    record_scores,
    set_config,
)


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def test_get_score_history_returns_scores_recorded_for_a_run():
    session = _session()

    record_scores(session, run_id="run-1", step=3, scores={"agent": 0.2, "model": 0.9})

    history = get_score_history(session, run_id="run-1")
    by_node = {row.node_name: row.risk_score for row in history}
    assert by_node == {"agent": 0.2, "model": 0.9}
    assert all(row.step == 3 for row in history)


def test_get_score_history_can_filter_by_node_name():
    session = _session()
    record_scores(session, run_id="run-1", step=0, scores={"agent": 0.1, "model": 0.5})

    history = get_score_history(session, run_id="run-1", node_name="model")

    assert [row.node_name for row in history] == ["model"]


def test_record_and_get_incidents_round_trip():
    session = _session()
    occurred_at = datetime(2026, 8, 28, tzinfo=timezone.utc)

    record_incident(
        session,
        run_id="run-1",
        node_name="model",
        incident_type="degradation",
        occurred_at=occurred_at,
        source="manual",
    )

    incidents = get_incidents(session, run_id="run-1")
    assert len(incidents) == 1
    assert incidents[0].node_name == "model"
    assert incidents[0].incident_type == "degradation"


def test_record_and_get_alert_history_round_trip():
    session = _session()

    record_alert(
        session, run_id="run-1", node_name="model", risk_score=0.95, message="model at risk", channel="webhook"
    )

    alerts = get_alert_history(session, run_id="run-1")
    assert len(alerts) == 1
    assert alerts[0].risk_score == 0.95
    assert alerts[0].channel == "webhook"


def test_get_latest_scores_returns_most_recent_score_per_node():
    session = _session()
    record_scores(session, run_id="run-1", step=0, scores={"agent": 0.1, "model": 0.2})
    record_scores(session, run_id="run-1", step=1, scores={"agent": 0.4})

    latest = get_latest_scores(session, run_id="run-1")

    assert latest == {"agent": 0.4, "model": 0.2}


def test_get_config_returns_default_when_unset():
    session = _session()

    assert get_config(session, "alerting_enabled", default="false") == "false"


def test_set_config_then_get_config_returns_updated_value():
    session = _session()

    set_config(session, "alerting_enabled", "true")
    set_config(session, "alerting_enabled", "false")

    assert get_config(session, "alerting_enabled") == "false"


def test_create_session_then_get_session_round_trips_the_token():
    session = _session()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    create_session(session, token="tok-123", expires_at=expires_at)

    found = get_session(session, "tok-123")
    assert found is not None
    assert found.token == "tok-123"


def test_get_session_returns_none_for_an_expired_token():
    session = _session()
    expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    create_session(session, token="tok-expired", expires_at=expires_at)

    assert get_session(session, "tok-expired") is None


def test_get_session_returns_none_for_an_unknown_token():
    session = _session()

    assert get_session(session, "no-such-token") is None


def test_delete_session_removes_it():
    session = _session()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    create_session(session, token="tok-123", expires_at=expires_at)

    delete_session(session, "tok-123")

    assert get_session(session, "tok-123") is None
