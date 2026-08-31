"""Integration seam: the retention background task actually starts on app startup
and shuts down cleanly on app shutdown (ADR 0004) -- the delete logic itself is
covered directly, without FastAPI, in tests/unit/test_retention.py; this only proves
the wiring in serving/api.py's lifespan doesn't hang or leave a dangling task."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cascaid.serving.api import create_app
from cascaid.storage.repository import init_db


def _session_factory() -> sessionmaker:
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    init_db(engine)
    return sessionmaker(bind=engine)


@pytest.mark.integration
def test_retention_task_starts_and_stops_cleanly_with_the_app_lifespan(tmp_path):
    app = create_app(model=None, store_dir=tmp_path, session_factory=_session_factory())

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
    # No assertion beyond "the with block exited without hanging or raising" --
    # that's the actual risk (task.cancel() not awaited correctly, or the
    # CancelledError propagating out of the lifespan and failing app shutdown).
