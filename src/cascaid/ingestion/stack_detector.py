"""Stack auto-detection: which orchestrator/model-gateway/vector-DB a pipeline uses (PRD section 4.5)."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass


def _module_is_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


VECTOR_DB_MODULES = ["pinecone", "weaviate", "pgvector"]


@dataclass(frozen=True)
class DetectedStack:
    orchestrator: str | None = None
    model_gateway: str | None = None
    vector_db: str | None = None


def detect_stack(is_available: Callable[[str], bool] = _module_is_available) -> DetectedStack:
    orchestrator = "langgraph" if is_available("langgraph") else None
    model_gateway = "litellm" if is_available("litellm") else None
    vector_db = next((m for m in VECTOR_DB_MODULES if is_available(m)), None)
    return DetectedStack(orchestrator=orchestrator, model_gateway=model_gateway, vector_db=vector_db)
