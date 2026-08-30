"""FastAPI wrapper around risk.py (PRD 5.2 Model Serving): loads the latest graph
snapshot for a run from the Graph Store and returns the GNN's per-node risk scores."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from cascaid.alerting.dispatch import send_webhook
from cascaid.alerting.rules import evaluate_risk
from cascaid.auth.dependency import make_require_auth
from cascaid.ingestion.graph_store import latest_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.risk import predict_risk
from cascaid.storage.repository import get_config, record_alert, record_scores


def _maybe_alert(session: Session, run_id: str, scores: dict[str, float], node_types: dict[str, str]) -> None:
    if get_config(session, "alerting_enabled", default="false") != "true":
        return
    webhook_url = get_config(session, "alert_webhook_url")
    if not webhook_url:
        return
    threshold = float(get_config(session, "alert_threshold", default="0.8"))
    for alert in evaluate_risk(run_id, scores, node_types, threshold):
        send_webhook(webhook_url, alert)
        record_alert(
            session,
            run_id=alert.run_id,
            node_name=alert.node_name,
            risk_score=alert.risk_score,
            message=alert.message,
            channel="webhook",
        )


def create_app(
    model: CascadeGNN | None,
    store_dir: str | Path,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    app = FastAPI()
    # No --database-url means no place to store a validatable session (see
    # cascaid.auth.dependency), so an ephemeral, unpersisted `cascaid serve` stays
    # open, same as it was before auth existed.
    auth_dependencies = [Depends(make_require_auth(session_factory))] if session_factory is not None else []

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/risk/{run_id}", dependencies=auth_dependencies)
    def risk(run_id: str):
        data = latest_snapshot(store_dir, run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        scores = predict_risk(model, data)
        if session_factory is not None:
            with session_factory() as session:
                record_scores(session, run_id=run_id, step=data.step, scores=scores)
                _maybe_alert(
                    session, run_id=run_id, scores=scores, node_types=dict(zip(data.node_order, data.node_types))
                )
        return {"run_id": run_id, "step": data.step, "scores": scores}

    return app
