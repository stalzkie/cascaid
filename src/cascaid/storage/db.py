"""Engine/session-factory construction from a DATABASE_URL (PRD 5.2 Storage).

Kept separate from repository.py so callers that already have a Session (e.g. a
test) never need to touch engine construction."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def make_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url))
