"""Webhook delivery for evaluated alerts (PRD 5.2 Alerting: webhook/Slack/PagerDuty).
Slack/PagerDuty are both webhook-shaped for v1, so this one function covers all three
-- a failed delivery never raises, since a down alert sink must not break risk serving."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from cascaid.alerting.rules import Alert


def send_webhook(url: str, alert: Alert, timeout: float = 5.0) -> bool:
    try:
        response = httpx.post(url, json=asdict(alert), timeout=timeout)
        return response.status_code < 400
    except httpx.HTTPError:
        return False
