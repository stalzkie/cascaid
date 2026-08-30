"""Webhook delivery for evaluated alerts (PRD 5.2 Alerting: webhook/Slack/PagerDuty).
Slack and PagerDuty each expect their own JSON envelope (a Slack incoming webhook
needs a top-level "text"/"blocks" key; PagerDuty's Events API v2 needs a
routing_key/event_action/payload envelope) -- format_slack_payload/
format_pagerduty_payload build those, while the plain "webhook" channel still posts
the raw Alert shape for a customer's own receiver. A failed delivery never raises,
since a down alert sink must not break risk serving."""

from __future__ import annotations

from dataclasses import asdict

import httpx
from sqlalchemy.orm import Session

from cascaid.alerting.rules import Alert
from cascaid.storage.repository import get_config


def enabled_webhook_url(session: Session) -> str | None:
    """None unless alerting is on AND a webhook is configured -- the shared gate
    every alert-firing call site (risk-threshold, drift, ...) checks before doing
    anything else, so "is alerting on and where does it go" has one place to read,
    even though what happens after differs per call site."""
    if get_config(session, "alerting_enabled", default="false") != "true":
        return None
    return get_config(session, "alert_webhook_url")


def alert_channel_config(session: Session) -> tuple[str, str | None]:
    """(channel, pagerduty_routing_key). channel defaults to "webhook" (the
    original generic-payload behavior) unless the user has explicitly configured
    "slack" or "pagerduty" via cascaid.alerting.configure."""
    channel = get_config(session, "alert_channel", default="webhook")
    routing_key = get_config(session, "alert_pagerduty_routing_key")
    return channel, routing_key


def format_slack_payload(alert: Alert) -> dict:
    return {"text": alert.message}


def format_pagerduty_payload(alert: Alert, routing_key: str) -> dict:
    return {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": f"{alert.run_id}:{alert.node_name}",
        "payload": {
            "summary": alert.message,
            "source": alert.run_id,
            "severity": "warning",
            "custom_details": {
                "node_name": alert.node_name,
                "node_type": alert.node_type,
                "risk_score": alert.risk_score,
            },
        },
    }


def send_webhook(
    url: str,
    alert: Alert,
    channel: str = "webhook",
    pagerduty_routing_key: str | None = None,
    timeout: float = 5.0,
) -> bool:
    if channel == "slack":
        body = format_slack_payload(alert)
    elif channel == "pagerduty":
        body = format_pagerduty_payload(alert, pagerduty_routing_key or "")
    else:
        body = asdict(alert)
    try:
        response = httpx.post(url, json=body, timeout=timeout)
        return response.status_code < 400
    except httpx.HTTPError:
        return False
