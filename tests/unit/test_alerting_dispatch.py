"""Unit seam: dispatch.py's pure payload formatters (PRD 5.2 Alerting: webhook/
Slack/PagerDuty). No network involved -- see test_alerting_dispatch.py under
tests/integration for send_webhook's channel-aware wiring against a real server."""

from cascaid.alerting.dispatch import format_pagerduty_payload, format_slack_payload
from cascaid.alerting.rules import Alert


def test_format_slack_payload_puts_the_alert_message_in_text():
    alert = Alert(run_id="run-1", node_name="store", node_type="vector_store", risk_score=0.9, message="at risk")
    assert format_slack_payload(alert) == {"text": "at risk"}


def test_format_pagerduty_payload_matches_events_api_v2_envelope():
    alert = Alert(run_id="run-1", node_name="store", node_type="vector_store", risk_score=0.9, message="at risk")
    payload = format_pagerduty_payload(alert, routing_key="rk-123")

    assert payload["routing_key"] == "rk-123"
    assert payload["event_action"] == "trigger"
    assert payload["dedup_key"] == "run-1:store"
    assert payload["payload"]["summary"] == "at risk"
    assert payload["payload"]["source"] == "run-1"
    assert payload["payload"]["custom_details"] == {
        "node_name": "store",
        "node_type": "vector_store",
        "risk_score": 0.9,
    }
