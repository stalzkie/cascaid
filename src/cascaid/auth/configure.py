"""Bootstrap the single self-hosted admin account (mirrors
cascaid.alerting.configure's shape):

    python -m cascaid.auth.configure --database-url ... \
        --set-username admin --set-password hunter2
"""

from __future__ import annotations

import argparse

from cascaid.auth.passwords import hash_password
from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import set_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", type=str, required=True)
    parser.add_argument("--set-username", type=str, default=None)
    parser.add_argument("--set-password", type=str, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    session_factory = make_session_factory(args.database_url)
    with session_factory() as session:
        if args.set_username is not None:
            set_config(session, "auth_username", args.set_username)
        if args.set_password is not None:
            set_config(session, "auth_password_hash", hash_password(args.set_password))


if __name__ == "__main__":
    main()
