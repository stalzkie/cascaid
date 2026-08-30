"""Pure prompt-building for LLM-generated risk explanations (PRD 7: "agent-checkout
is at elevated risk because its vector-store dependency has shown rising p99
retrieval latency"). No LLM call, no I/O -- see cascaid.explain.client for that half.
"""

from __future__ import annotations

from cascaid.ingestion.schema import FEATURE_NAMES, NUM_FEATURES


def node_feature_dict(data, node_name: str) -> dict[str, float]:
    """The first NUM_FEATURES columns of data.x's row for node_name, in
    FEATURE_NAMES order -- the rest of the row is the node-type one-hot, which
    this deliberately ignores (node_type is passed separately, already
    human-readable, by whoever calls this)."""
    idx = data.node_order.index(node_name)
    row = data.x[idx]
    row = row.tolist() if hasattr(row, "tolist") else row
    return dict(zip(FEATURE_NAMES, (float(v) for v in row[:NUM_FEATURES])))


def callees_of(node_name: str, edges: list[tuple[str, str]]) -> list[str]:
    """Nodes node_name directly calls -- its downstream dependencies, the
    direction PRD 7's own example explanation points at ("its vector-store
    dependency"), not its callers."""
    return [callee for caller, callee in edges if caller == node_name]


def build_explanation_prompt(
    node_name: str,
    node_type: str,
    risk_score: float,
    own_features: dict[str, float],
    neighbor_features: dict[str, dict[str, float]],
) -> str:
    lines = [
        "You are explaining a cascade-risk score from an AI pipeline monitoring "
        "system to an on-call engineer. Be concise (2-3 sentences), plain-English, "
        "and specific about which dependency and which metric is driving the risk.",
        "",
        f"Node: {node_name} (type: {node_type})",
        f"Current risk score: {risk_score:.2f} (0=healthy, 1=critical)",
        f"Its own current metrics: {own_features}",
    ]
    if neighbor_features:
        lines.append("Metrics for the nodes it directly depends on:")
        for name, features in neighbor_features.items():
            lines.append(f"  - {name}: {features}")
    else:
        lines.append("It has no direct downstream dependencies.")
    lines.append("")
    lines.append("Explain why this node is at risk, referencing the specific metric(s) responsible.")
    return "\n".join(lines)
