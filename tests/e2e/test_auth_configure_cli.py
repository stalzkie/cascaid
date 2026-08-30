"""E2E seam: the real CLI a user runs to bootstrap the single self-hosted admin
account (mirrors cascaid.alerting.configure)."""

import sys

import pytest

import cascaid.auth.configure as configure_cli
from cascaid.auth.passwords import verify_password
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_config, init_db


@pytest.mark.e2e
def test_configure_cli_sets_username_and_a_verifiable_password_hash(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "configure",
            "--database-url",
            database_url,
            "--set-username",
            "admin",
            "--set-password",
            "hunter2",
        ],
    )
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "auth_username") == "admin"
        hashed = get_config(session, "auth_password_hash")
        assert hashed is not None
        assert verify_password("hunter2", hashed) is True
        assert verify_password("wrong", hashed) is False
