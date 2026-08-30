"""MCP tool logic (PRD 7: "any agent ... can query 'what's the current cascade risk
on our RAG pipeline' as a direct tool call"), kept free of the MCP SDK so it's
unit-testable without a stdio transport. Current risk is whatever Model Serving
last persisted -- same source of truth as the dashboard API's /pipeline endpoint,
no model loaded here."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from cascaid.dashboard.views import pipeline_view


def get_cascade_risk(store_dir: str | Path, session_factory: sessionmaker[Session], run_id: str) -> dict | None:
    with session_factory() as session:
        return pipeline_view(store_dir, session, run_id)
