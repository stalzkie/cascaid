"""FastAPI wrapper around views.py (PRD 5.2 Dashboard API): serves the risk graph
and track record to the frontend/Grafana panel/MCP server (Section 4.7)."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from cascaid.auth.dependency import make_require_auth
from cascaid.auth.passwords import verify_password
from cascaid.dashboard.grafana import grafana_query, grafana_search
from cascaid.dashboard.views import list_runs_view, pipeline_view, track_record_view
from cascaid.storage.repository import create_session, delete_session, get_config

SESSION_LIFETIME = timedelta(hours=24)


class LoginRequest(BaseModel):
    username: str
    password: str


class GrafanaTarget(BaseModel):
    target: str


class GrafanaQueryRequest(BaseModel):
    targets: list[GrafanaTarget]


def create_app(store_dir: str | Path, session_factory: sessionmaker[Session]) -> FastAPI:
    app = FastAPI()
    # Self-hosted-first (PRD 5.2 Deployment): the frontend and this API run as
    # separate containers/origins within the customer's own VPC, not a multi-tenant
    # SaaS boundary, so an open CORS policy here doesn't cross a trust boundary.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    require_auth = make_require_auth(session_factory)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/auth/login")
    def login(body: LoginRequest):
        with session_factory() as session:
            expected_username = get_config(session, "auth_username")
            expected_hash = get_config(session, "auth_password_hash")
            if (
                expected_username is None
                or expected_hash is None
                or body.username != expected_username
                or not verify_password(body.password, expected_hash)
            ):
                raise HTTPException(status_code=401, detail="invalid username or password")
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now(timezone.utc) + SESSION_LIFETIME
            create_session(session, token=token, expires_at=expires_at)
        return {"token": token, "expires_at": expires_at.isoformat()}

    @app.post("/auth/logout", dependencies=[Depends(require_auth)])
    def logout(authorization: str = Header(default="")):
        _, _, token = authorization.partition(" ")
        with session_factory() as session:
            delete_session(session, token)
        return {"status": "ok"}

    @app.get("/runs", dependencies=[Depends(require_auth)])
    def runs():
        return {"run_ids": list_runs_view(store_dir)}

    @app.get("/pipeline/{run_id}", dependencies=[Depends(require_auth)])
    def pipeline(run_id: str):
        with session_factory() as session:
            view = pipeline_view(store_dir, session, run_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        return view

    @app.get("/track-record/{run_id}", dependencies=[Depends(require_auth)])
    def track_record(run_id: str):
        with session_factory() as session:
            return track_record_view(session, run_id)

    # SimPod-json-datasource/Infinity-compatible endpoints (PRD 4.7): lets a Grafana
    # panel query Cascaid via an existing community JSON datasource plugin, no custom
    # Grafana plugin to build/sign. See dashboard/grafana.py.
    @app.get("/grafana/", dependencies=[Depends(require_auth)])
    def grafana_test_connection_route():
        return {"status": "ok"}

    @app.post("/grafana/search", dependencies=[Depends(require_auth)])
    def grafana_search_route():
        return grafana_search(store_dir)

    @app.post("/grafana/query", dependencies=[Depends(require_auth)])
    def grafana_query_route(body: GrafanaQueryRequest):
        try:
            return grafana_query(session_factory, [t.target for t in body.targets])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
