"""FastAPI wrapper around risk.py (PRD 5.2 Model Serving): loads the latest graph
snapshot for a run from the Graph Store and returns the GNN's per-node risk scores."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from cascaid.alerting.dispatch import alert_channel_config, enabled_webhook_url, send_webhook
from cascaid.alerting.rules import evaluate_risk
from cascaid.auth.dependency import make_require_auth
from cascaid.explain.client import generate_explanation
from cascaid.explain.prompt import build_explanation_prompt, callees_of, node_feature_dict
from cascaid.ingestion.graph_store import latest_snapshot
from cascaid.models.gnn import CascadeGNN
from cascaid.serving.risk import predict_risk
from cascaid.storage.repository import get_config, record_alert, record_scores


def _maybe_alert(session: Session, run_id: str, scores: dict[str, float], node_types: dict[str, str]) -> None:
    webhook_url = enabled_webhook_url(session)
    if not webhook_url:
        return
    channel, routing_key = alert_channel_config(session)
    threshold = float(get_config(session, "alert_threshold", default="0.8"))
    for alert in evaluate_risk(run_id, scores, node_types, threshold):
        send_webhook(webhook_url, alert, channel=channel, pagerduty_routing_key=routing_key)
        record_alert(
            session,
            run_id=alert.run_id,
            node_name=alert.node_name,
            risk_score=alert.risk_score,
            message=alert.message,
            channel=channel,
        )


def create_app(
    model: CascadeGNN | None,
    store_dir: str | Path,
    session_factory: sessionmaker[Session],
) -> FastAPI:
    app = FastAPI()
    # Self-hosted-first (PRD 5.2 Deployment): the frontend and this API run as
    # separate containers/origins within the customer's own VPC, not a multi-tenant
    # SaaS boundary, so an open CORS policy here doesn't cross a trust boundary.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    require_auth = make_require_auth(session_factory)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/risk/{run_id}", dependencies=[Depends(require_auth)])
    def risk(run_id: str):
        data = latest_snapshot(store_dir, run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        scores = predict_risk(model, data)
        with session_factory() as session:
            record_scores(session, run_id=run_id, step=data.step, scores=scores)
            _maybe_alert(session, run_id=run_id, scores=scores, node_types=dict(zip(data.node_order, data.node_types)))
        return {"run_id": run_id, "step": data.step, "scores": scores}

    @app.get("/risk/{run_id}/explain/{node_name}", dependencies=[Depends(require_auth)])
    def explain(run_id: str, node_name: str):
        with session_factory() as session:
            if get_config(session, "llm_explanations_enabled", default="false") != "true":
                raise HTTPException(status_code=404, detail="LLM risk explanations are not enabled")
            base_url = get_config(session, "llm_base_url")
            api_key = get_config(session, "llm_api_key", default="")
            model_name = get_config(session, "llm_model")

        data = latest_snapshot(store_dir, run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no snapshot found for run_id={run_id!r}")
        if node_name not in data.node_order:
            raise HTTPException(status_code=404, detail=f"no node {node_name!r} in run_id={run_id!r}")

        scores = predict_risk(model, data)
        node_types = dict(zip(data.node_order, data.node_types))
        own_features = node_feature_dict(data, node_name)
        neighbor_features = {
            n: {"node_type": node_types[n], **node_feature_dict(data, n)} for n in callees_of(node_name, data.edges)
        }
        prompt = build_explanation_prompt(
            node_name, node_types[node_name], scores[node_name], own_features, neighbor_features
        )

        explanation = generate_explanation(prompt, base_url=base_url, api_key=api_key, model=model_name)
        if explanation is None:
            raise HTTPException(status_code=503, detail="LLM explanation endpoint is unreachable or misconfigured")

        return {"run_id": run_id, "node_name": node_name, "risk_score": scores[node_name], "explanation": explanation}

    return app
