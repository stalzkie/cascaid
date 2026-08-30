"""E2E seam: the real CLI a user runs to opt in to alerting (PRD 4.6 progressive
trust -- alerting stays off until explicitly enabled)."""

import sys

import pytest

import cascaid.alerting.configure as configure_cli
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_config, init_db


@pytest.mark.e2e
def test_configure_cli_enables_alerting_with_threshold_and_webhook(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "configure",
            "--database-url",
            database_url,
            "--enable",
            "--threshold",
            "0.75",
            "--webhook-url",
            "https://hooks.example.com/cascaid",
        ],
    )
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "alerting_enabled") == "true"
        assert get_config(session, "alert_threshold") == "0.75"
        assert get_config(session, "alert_webhook_url") == "https://hooks.example.com/cascaid"


@pytest.mark.e2e
def test_configure_cli_sets_slack_channel(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "configure",
            "--database-url",
            database_url,
            "--enable",
            "--webhook-url",
            "https://hooks.slack.com/services/xyz",
            "--channel",
            "slack",
        ],
    )
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "alert_channel") == "slack"


@pytest.mark.e2e
def test_configure_cli_sets_pagerduty_channel_and_routing_key(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "configure",
            "--database-url",
            database_url,
            "--enable",
            "--webhook-url",
            "https://events.pagerduty.com/v2/enqueue",
            "--channel",
            "pagerduty",
            "--pagerduty-routing-key",
            "rk-123",
        ],
    )
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "alert_channel") == "pagerduty"
        assert get_config(session, "alert_pagerduty_routing_key") == "rk-123"


@pytest.mark.e2e
def test_configure_cli_disable_turns_alerting_back_off(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))
    monkeypatch.setattr(sys, "argv", ["configure", "--database-url", database_url, "--enable"])
    configure_cli.main()

    monkeypatch.setattr(sys, "argv", ["configure", "--database-url", database_url, "--disable"])
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "alerting_enabled") == "false"
