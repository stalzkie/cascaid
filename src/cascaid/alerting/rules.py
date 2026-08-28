"""Threshold-based alert evaluation (PRD 3.5, 5.2 Alerting): turns per-node risk
scores into AI-specific alert copy naming the node and the likely downstream impact.
Off by default in the caller (cascaid.serving.api) -- this module has no concept of
enabled/disabled, it just answers "what would fire given this threshold"."""

from __future__ import annotations

from dataclasses import dataclass

_IMPACT_BY_NODE_TYPE = {
    "vector_store": "expect downstream generation quality to degrade soon",
    "model_endpoint": "expect latency and fallback-quality impact",
    "agent": "its downstream tool/model calls may start failing",
    "tool": "downstream steps depending on it may start failing",
}

_LABEL_BY_NODE_TYPE = {
    "vector_store": "Vector store",
    "model_endpoint": "Model endpoint",
    "agent": "Agent",
    "tool": "Tool",
}


@dataclass(frozen=True)
class Alert:
    run_id: str
    node_name: str
    node_type: str
    risk_score: float
    message: str


def _message(node_name: str, node_type: str, risk_score: float) -> str:
    label = _LABEL_BY_NODE_TYPE.get(node_type, "Node")
    impact = _IMPACT_BY_NODE_TYPE.get(node_type, "expect downstream impact")
    return f"{label} '{node_name}' is at elevated cascade risk (score={risk_score:.2f}) -- {impact}."


def evaluate_risk(
    run_id: str,
    node_scores: dict[str, float],
    node_types: dict[str, str],
    threshold: float,
) -> list[Alert]:
    return [
        Alert(
            run_id=run_id,
            node_name=node,
            node_type=node_types[node],
            risk_score=score,
            message=_message(node, node_types[node], score),
        )
        for node, score in node_scores.items()
        if score >= threshold
    ]
