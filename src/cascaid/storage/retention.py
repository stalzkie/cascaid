"""Built-in retention for score/incident/alert history (ADR 0004): a periodic-delete
job, not TimescaleDB or a documented customer-side cron -- see the ADR for why. No new
container, no new table: this runs as a background task inside the existing `serve`
process (wired into serving/api.py's app lifespan), and the retention window itself is
just another `Config` key/value entry.

DEFAULT_RETENTION_DAYS is deliberately conservative (90 days, roughly a quarter): ADR
0004 flags picking too short a default as a real risk, since score/incident history is
also what drift comparison (cascaid.drift) reasons over, and deleting it prematurely
degrades that rather than just saving disk space. A customer who wants a shorter window
can set it explicitly via CASCAID_RETENTION_DAYS or the `retention_days` Config key.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from cascaid.storage.models import AlertHistory, IncidentLabel, ScoreHistory
from cascaid.storage.repository import get_config

DEFAULT_RETENTION_DAYS = 90
DEFAULT_CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # once a day is plenty for a day-granularity window

logger = logging.getLogger(__name__)


def _retention_days(session: Session) -> int:
    default = os.environ.get("CASCAID_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
    return int(get_config(session, "retention_days", default=default))


def delete_expired_history(session: Session, retention_days: int) -> dict[str, int]:
    """Deletes rows older than retention_days from the three history tables that grow
    without bound, and commits. Returns how many rows were deleted from each, for
    logging/observability -- not used for control flow."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = {
        "score_history": session.execute(delete(ScoreHistory).where(ScoreHistory.predicted_at < cutoff)).rowcount,
        "incident_labels": session.execute(delete(IncidentLabel).where(IncidentLabel.occurred_at < cutoff)).rowcount,
        "alert_history": session.execute(delete(AlertHistory).where(AlertHistory.sent_at < cutoff)).rowcount,
    }
    session.commit()
    return deleted


async def run_retention_loop(
    session_factory: sessionmaker[Session],
    *,
    check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS,
) -> None:
    """Runs forever (intended to be wrapped in an asyncio.Task and cancelled on app
    shutdown -- see serving/api.py's lifespan). retention_days is re-read from Config
    every cycle rather than captured once, so a customer changing it via
    `set_config`/a future configure CLI takes effect without restarting the process.

    A single failed cycle (e.g. a transient DB connectivity blip) is logged and
    retried next cycle rather than killing the loop -- retention silently stopping
    forever on one bad cycle is a worse failure mode than one skipped cycle."""
    while True:
        await asyncio.sleep(check_interval_seconds)
        try:
            with session_factory() as session:
                retention_days = _retention_days(session)
                deleted = delete_expired_history(session, retention_days)
                logger.info("retention: deleted %s (retention_days=%d)", deleted, retention_days)
        except Exception:
            logger.exception("retention cycle failed, will retry next cycle")
