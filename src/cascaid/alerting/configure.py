"""Opt-in switch for alerting (PRD 4.6 progressive trust: observe-only until the
user explicitly enables it):

    python -m cascaid.alerting.configure --database-url ... --enable \
        --threshold 0.8 --webhook-url https://hooks.example.com/cascaid
    python -m cascaid.alerting.configure --database-url ... --disable

Slack: --webhook-url is the Slack incoming-webhook URL, --channel slack.
PagerDuty: --webhook-url is https://events.pagerduty.com/v2/enqueue,
--channel pagerduty, and --pagerduty-routing-key is the service's integration key.
"""

from __future__ import annotations

import argparse

from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import set_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--webhook-url", type=str, default=None)
    parser.add_argument("--channel", type=str, choices=["webhook", "slack", "pagerduty"], default=None)
    parser.add_argument("--pagerduty-routing-key", type=str, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    session_factory = make_session_factory(args.database_url)
    with session_factory() as session:
        if args.enable:
            set_config(session, "alerting_enabled", "true")
        if args.disable:
            set_config(session, "alerting_enabled", "false")
        if args.threshold is not None:
            set_config(session, "alert_threshold", str(args.threshold))
        if args.webhook_url is not None:
            set_config(session, "alert_webhook_url", args.webhook_url)
        if args.channel is not None:
            set_config(session, "alert_channel", args.channel)
        if args.pagerduty_routing_key is not None:
            set_config(session, "alert_pagerduty_routing_key", args.pagerduty_routing_key)


if __name__ == "__main__":
    main()
