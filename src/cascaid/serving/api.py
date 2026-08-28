"""FastAPI wrapper around risk.py (PRD 5.2 Model Serving): loads the latest graph
snapshot for a run from the Graph Store and returns the GNN's per-node risk scores."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from cascaid.ingestion.graph_store import latest_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.risk import predict_risk
from cascaid.storage.repository import record_scores


def create_app(
    model: CascadeGNN | None,
    store_dir: str | Path,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/risk/{run_id}")
    def risk(run_id: str):
        data = latest_snapshot(store_dir, run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        scores = predict_risk(model, data)
        if session_factory is not None:
            with session_factory() as session:
                record_scores(session, run_id=run_id, step=data.step, scores=scores)
        return {"run_id": run_id, "step": data.step, "scores": scores}

    return app
