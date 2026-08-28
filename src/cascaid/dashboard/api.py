"""FastAPI wrapper around views.py (PRD 5.2 Dashboard API): serves the risk graph
and track record to the frontend/Grafana panel/MCP server (Section 4.7)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from cascaid.dashboard.views import list_runs_view, pipeline_view, track_record_view


def create_app(store_dir: str | Path, session_factory: sessionmaker[Session]) -> FastAPI:
    app = FastAPI()
    # Self-hosted-first (PRD 5.2 Deployment): the frontend and this API run as
    # separate containers/origins within the customer's own VPC, not a multi-tenant
    # SaaS boundary, so an open CORS policy here doesn't cross a trust boundary.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/runs")
    def runs():
        return {"run_ids": list_runs_view(store_dir)}

    @app.get("/pipeline/{run_id}")
    def pipeline(run_id: str):
        with session_factory() as session:
            view = pipeline_view(store_dir, session, run_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        return view

    @app.get("/track-record/{run_id}")
    def track_record(run_id: str):
        with session_factory() as session:
            return track_record_view(session, run_id)

    return app
