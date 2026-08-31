"""Encrypts sensitive Config values at rest (ADR 0005) -- concretely, `llm_api_key`
(PRD 7's bring-your-own LLM endpoint, see cascaid.explain.configure/client). Deliberate
seam choice: dedicated set_secret_config/get_secret_config functions, not a hidden
allowlist inside repository.get_config/set_config -- an explicit function name at each
call site is harder to silently get wrong than a magic-string allowlist a future
sensitive Config key could easily miss, and it doesn't change behavior for the many
existing set_config/get_config callers that aren't secrets.

CASCAID_CONFIG_ENCRYPTION_KEY is required to use these (not `Config` itself, since the
whole point is keeping it out of the database) -- a Fernet key, e.g. generated via
`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
Losing or rotating this key loses the ability to decrypt whatever was already stored
under it (not the row itself, which is still there, just unreadable) -- reconfiguring
the secret (re-running `cascaid.explain.configure --api-key ...`) recovers from that,
same as if the key had simply been mistyped. A missing key or a value that fails to
decrypt under the configured key both raise a clear, specific error rather than
returning garbage or silently falling back to an empty string.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from cascaid.storage.repository import get_config, set_config

_ENV_VAR = "CASCAID_CONFIG_ENCRYPTION_KEY"


class MissingEncryptionKeyError(RuntimeError):
    def __init__(self):
        super().__init__(
            f"{_ENV_VAR} is not set -- required to read or write an encrypted Config value "
            "(e.g. llm_api_key). Generate one with: "
            'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )


class SecretDecryptionError(RuntimeError):
    def __init__(self, key: str):
        super().__init__(
            f"Config key {key!r} could not be decrypted with the current {_ENV_VAR} -- it was "
            "likely encrypted under a different key. Reconfigure it (e.g. re-run "
            "`python -m cascaid.explain.configure --api-key ...`) to recover."
        )


def _fernet() -> Fernet:
    raw_key = os.environ.get(_ENV_VAR)
    if not raw_key:
        raise MissingEncryptionKeyError()
    return Fernet(raw_key.encode())


def set_secret_config(session: Session, key: str, value: str) -> None:
    token = _fernet().encrypt(value.encode()).decode()
    set_config(session, key, token)


def get_secret_config(session: Session, key: str, default: str | None = None) -> str | None:
    token = get_config(session, key)
    if token is None:
        return default
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise SecretDecryptionError(key) from exc
