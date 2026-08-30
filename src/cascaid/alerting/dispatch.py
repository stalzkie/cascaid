"""Webhook delivery for evaluated alerts (PRD 5.2 Alerting: webhook/Slack/PagerDuty).
Slack/PagerDuty are both webhook-shaped for v1, so this one function covers all three
-- a failed delivery never raises, since a down alert sink must not break risk serving."""

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


def send_webhook(url: str, alert: Alert, timeout: float = 5.0) -> bool:
    try:
        response = httpx.post(url, json=asdict(alert), timeout=timeout)
        return response.status_code < 400
    except httpx.HTTPError:
        return False
