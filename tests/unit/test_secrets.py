import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from cascaid.storage.repository import get_config, init_db
from cascaid.storage.secrets import (
    MissingEncryptionKeyError,
    SecretDecryptionError,
    get_secret_config,
    set_secret_config,
)


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return Session(engine)


def test_set_and_get_secret_config_round_trips(monkeypatch):
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())
    session = _session()

    set_secret_config(session, "llm_api_key", "sk-real-secret")

    assert get_secret_config(session, "llm_api_key") == "sk-real-secret"


def test_set_secret_config_does_not_store_plaintext(monkeypatch):
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())
    session = _session()

    set_secret_config(session, "llm_api_key", "sk-real-secret")

    raw = get_config(session, "llm_api_key")
    assert raw != "sk-real-secret"
    assert "sk-real-secret" not in raw


def test_get_secret_config_returns_default_when_unset(monkeypatch):
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())
    session = _session()

    assert get_secret_config(session, "llm_api_key", default="") == ""


def test_set_secret_config_raises_a_clear_error_without_an_encryption_key(monkeypatch):
    monkeypatch.delenv("CASCAID_CONFIG_ENCRYPTION_KEY", raising=False)
    session = _session()

    with pytest.raises(MissingEncryptionKeyError):
        set_secret_config(session, "llm_api_key", "sk-real-secret")


def test_get_secret_config_raises_a_clear_error_under_a_different_key(monkeypatch):
    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())
    session = _session()
    set_secret_config(session, "llm_api_key", "sk-real-secret")

    monkeypatch.setenv("CASCAID_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())

    with pytest.raises(SecretDecryptionError):
        get_secret_config(session, "llm_api_key")
