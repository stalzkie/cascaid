"""FastAPI auth dependency shared by dashboard/api.py and serving/api.py: validates
a bearer token against the AuthSession table (see storage/repository.py)."""

from __future__ import annotations

from fastapi import Header, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from cascaid.storage.repository import get_session


def make_require_auth(session_factory: sessionmaker[Session]):
    def require_auth(authorization: str = Header(default="")) -> None:
        scheme, _, token = authorization.partition(" ")
        if scheme != "Bearer" or not token:
            raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
        with session_factory() as session:
            if get_session(session, token) is None:
                raise HTTPException(status_code=401, detail="invalid or expired session")

    return require_auth
