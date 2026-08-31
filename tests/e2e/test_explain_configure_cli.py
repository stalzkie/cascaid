"""E2E seam: the real CLI a user runs to opt in to LLM risk explanations (PRD 7,
off by default, bring-your-own-endpoint)."""

import sys

import pytest
from cryptography.fernet import Fernet

import cascaid.explain.configure as configure_cli
from cascaid.storage.db import get_engine, make_session_factory
from cascaid.storage.repository import get_config, init_db
from cascaid.storage.secrets import get_secret_config


@pytest.mark.e2e
def test_configure_cli_enables_explanations_with_a_byo_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())
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
            "--base-url",
            "http://localhost:11434/v1",
            "--api-key",
            "unused",
            "--model",
            "llama3",
        ],
    )
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "llm_explanations_enabled") == "true"
        assert get_config(session, "llm_base_url") == "http://localhost:11434/v1"
        assert get_config(session, "llm_model") == "llama3"
        # ADR 0005: llm_api_key is encrypted at rest -- the raw Config row must NOT
        # be the plaintext value, but it must round-trip correctly through the
        # secret-config seam.
        assert get_config(session, "llm_api_key") != "unused"
        assert get_secret_config(session, "llm_api_key") == "unused"


@pytest.mark.e2e
def test_configure_cli_disable_turns_explanations_back_off(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'cascaid.db'}"
    init_db(get_engine(database_url))
    monkeypatch.setattr(sys, "argv", ["configure", "--database-url", database_url, "--enable"])
    configure_cli.main()

    monkeypatch.setattr(sys, "argv", ["configure", "--database-url", database_url, "--disable"])
    configure_cli.main()

    with make_session_factory(database_url)() as session:
        assert get_config(session, "llm_explanations_enabled") == "false"
