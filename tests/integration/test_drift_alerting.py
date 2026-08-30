"""Integration seam: cascaid.drift._maybe_alert against a real sqlite DB and a real
HTTP webhook, reusing the existing alerting Config/AlertHistory plumbing."""

import pytest

from cascaid.drift import _maybe_alert
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_alert_history, init_db, set_config


@pytest.mark.integration
def test_maybe_alert_fires_webhook_and_records_history_when_enabled(tmp_path, httpserver):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))
    session_factory = make_session_factory(database_url)
    with session_factory() as session:
        set_config(session, "alerting_enabled", "true")
        set_config(session, "alert_webhook_url", httpserver.url_for("/hook"))
    httpserver.expect_request("/hook", method="POST").respond_with_json({"ok": True})

    _maybe_alert("run-1", {"latency_ms": 0.45}, database_url)

    with session_factory() as session:
        history = get_alert_history(session, run_id="run-1")
    assert len(history) == 1
    assert history[0].node_name == "latency_ms"
    assert history[0].channel == "webhook"


@pytest.mark.integration
def test_maybe_alert_does_nothing_when_alerting_disabled(tmp_path, httpserver):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))
    session_factory = make_session_factory(database_url)
    with session_factory() as session:
        set_config(session, "alert_webhook_url", httpserver.url_for("/hook"))

    _maybe_alert("run-1", {"latency_ms": 0.45}, database_url)

    with session_factory() as session:
        history = get_alert_history(session, run_id="run-1")
    assert history == []
