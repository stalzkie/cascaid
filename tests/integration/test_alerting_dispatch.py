"""Integration seam: send_webhook against a real local HTTP server (not a mock),
matching this repo's convention of exercising real components (PRD 5.2 Alerting)."""

import pytest

from cascaid.alerting.dispatch import send_webhook
from cascaid.alerting.rules import Alert


@pytest.mark.integration
def test_send_webhook_posts_alert_payload_to_configured_url(httpserver):
    alert = Alert(run_id="run-1", node_name="store", node_type="vector_store", risk_score=0.9, message="at risk")
    httpserver.expect_request(
        "/hook",
        method="POST",
        json={
            "run_id": "run-1",
            "node_name": "store",
            "node_type": "vector_store",
            "risk_score": 0.9,
            "message": "at risk",
        },
    ).respond_with_json({"ok": True})

    ok = send_webhook(httpserver.url_for("/hook"), alert)

    assert ok is True


@pytest.mark.integration
def test_send_webhook_returns_false_on_server_error(httpserver):
    alert = Alert(run_id="run-1", node_name="store", node_type="vector_store", risk_score=0.9, message="at risk")
    httpserver.expect_request("/hook", method="POST").respond_with_data(status=500)

    ok = send_webhook(httpserver.url_for("/hook"), alert)

    assert ok is False


@pytest.mark.integration
def test_send_webhook_returns_false_when_unreachable():
    alert = Alert(run_id="run-1", node_name="store", node_type="vector_store", risk_score=0.9, message="at risk")

    ok = send_webhook("http://127.0.0.1:1/hook", alert, timeout=0.5)

    assert ok is False
