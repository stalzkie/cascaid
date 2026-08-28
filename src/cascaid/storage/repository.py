"""Public storage seam (PRD 5.2): record/query score history, incident labels, and
alert history, plus a flat key/value config store. Callers hand in a live Session
(alerting, the dashboard API, and the CLI each own their own engine/session lifecycle)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from cascaid.storage.models import AlertHistory, Base, Config, IncidentLabel, ScoreHistory


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def record_scores(session: Session, run_id: str, step: int, scores: dict[str, float]) -> None:
    session.add_all(
        ScoreHistory(run_id=run_id, step=step, node_name=node, risk_score=score) for node, score in scores.items()
    )
    session.commit()


def get_score_history(session: Session, run_id: str, node_name: str | None = None) -> list[ScoreHistory]:
    stmt = select(ScoreHistory).where(ScoreHistory.run_id == run_id)
    if node_name is not None:
        stmt = stmt.where(ScoreHistory.node_name == node_name)
    return list(session.scalars(stmt.order_by(ScoreHistory.step)))


def get_latest_scores(session: Session, run_id: str) -> dict[str, float]:
    latest: dict[str, float] = {}
    for row in get_score_history(session, run_id):
        latest[row.node_name] = row.risk_score
    return latest


def record_incident(
    session: Session, run_id: str, node_name: str, incident_type: str, occurred_at: datetime, source: str
) -> IncidentLabel:
    incident = IncidentLabel(
        run_id=run_id, node_name=node_name, incident_type=incident_type, occurred_at=occurred_at, source=source
    )
    session.add(incident)
    session.commit()
    return incident


def get_incidents(session: Session, run_id: str | None = None) -> list[IncidentLabel]:
    stmt = select(IncidentLabel)
    if run_id is not None:
        stmt = stmt.where(IncidentLabel.run_id == run_id)
    return list(session.scalars(stmt.order_by(IncidentLabel.occurred_at)))


def record_alert(
    session: Session, run_id: str, node_name: str, risk_score: float, message: str, channel: str
) -> AlertHistory:
    alert = AlertHistory(run_id=run_id, node_name=node_name, risk_score=risk_score, message=message, channel=channel)
    session.add(alert)
    session.commit()
    return alert


def get_alert_history(session: Session, run_id: str | None = None) -> list[AlertHistory]:
    stmt = select(AlertHistory)
    if run_id is not None:
        stmt = stmt.where(AlertHistory.run_id == run_id)
    return list(session.scalars(stmt.order_by(AlertHistory.sent_at)))


def get_config(session: Session, key: str, default: str | None = None) -> str | None:
    row = session.get(Config, key)
    return row.value if row is not None else default


def set_config(session: Session, key: str, value: str) -> None:
    row = session.get(Config, key)
    if row is None:
        session.add(Config(key=key, value=value))
    else:
        row.value = value
    session.commit()
