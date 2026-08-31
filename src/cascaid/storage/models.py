"""PRD 5.2 Storage: score history, incident labels, alert history, configuration.

Postgres is the target; SQLite is used for fast unit tests against the same schema.
Retention (expiring old score_history/incident_labels/alert_history rows) is a
built-in periodic-delete job, not TimescaleDB hypertables -- see
docs/adr/0004-retention-via-builtin-periodic-delete-not-timescale.md for why.
Schema changes go through Alembic (see docs/Migrations.md and
docs/adr/0003-adopt-alembic-for-schema-migrations.md); `init_db`/`create_all` below
stays for fresh installs and tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ScoreHistory(Base):
    __tablename__ = "score_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    step: Mapped[int] = mapped_column(Integer)
    node_name: Mapped[str] = mapped_column(String, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class IncidentLabel(Base):
    __tablename__ = "incident_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    node_name: Mapped[str] = mapped_column(String, index=True)
    incident_type: Mapped[str] = mapped_column(String)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String)


class AlertHistory(Base):
    __tablename__ = "alert_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    node_name: Mapped[str] = mapped_column(String, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(String)
    channel: Mapped[str] = mapped_column(String)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Config(Base):
    __tablename__ = "config"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


class AuthSession(Base):
    __tablename__ = "auth_session"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
