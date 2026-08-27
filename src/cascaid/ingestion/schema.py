"""Typed node/edge schema for a pipeline topology snapshot (PRD section 5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    MODEL_ENDPOINT = "model_endpoint"
    VECTOR_STORE = "vector_store"


NODE_TYPE_ORDER = [
    NodeType.AGENT,
    NodeType.TOOL,
    NodeType.MODEL_ENDPOINT,
    NodeType.VECTOR_STORE,
]


@dataclass(frozen=True)
class CallEvent:
    """One observed call (LiteLLM/gateway log line, vector DB query, tool invocation)."""

    run_id: str
    scenario: str
    step: int
    caller: str
    callee: str
    caller_type: NodeType
    callee_type: NodeType
    latency_ms: float
    error: bool
    retried: bool
    token_cost: float

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["caller_type"] = self.caller_type.value
        d["callee_type"] = self.callee_type.value
        return d

    @staticmethod
    def from_json(d: dict) -> "CallEvent":
        d = dict(d)
        d["caller_type"] = NodeType(d["caller_type"])
        d["callee_type"] = NodeType(d["callee_type"])
        return CallEvent(**d)


# Edge feature vector order used everywhere a raw (latency, error_rate, retry_rate,
# token_cost) tuple is turned into an array. Node features reuse the same order
# (aggregated over a node's incoming edges) so x and edge_attr share one convention.
FEATURE_NAMES = ["latency_ms", "error_rate", "retry_rate", "token_cost"]
NUM_FEATURES = len(FEATURE_NAMES)

# Nominal, healthy-but-idle defaults for an edge with no observed history yet
# (e.g. the fallback-model edge before any fallback has ever triggered).
NOMINAL_DEFAULTS = {
    "control": (5.0, 0.0, 0.0, 0.0),
    "vector_store": (60.0, 0.02, 0.0, 0.0),
    "model_endpoint": (180.0, 0.03, 0.0, 0.02),
}
