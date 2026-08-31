"""Stack auto-detection: which orchestrators/model-gateway/vector-DB a pipeline uses
(PRD section 4.5).

Orchestrators are detected independently, not exclusively: cascaid's own demo
pipeline needs `langgraph` installed, so it's always importable in any real install
of cascaid, whether or not the customer's actual app uses it. Picking a single
"winner" would make every orchestrator other than langgraph permanently
undetectable. Each orchestrator's instrumentation only fires when the target app
actually calls into that framework's API, so being wired up when unused is inert.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field

ORCHESTRATOR_MODULES = ["langgraph", "crewai"]
VECTOR_DB_MODULES = ["pinecone", "weaviate", "pgvector"]
# Direct model-provider SDKs, detected independently like ORCHESTRATOR_MODULES, not
# exclusively like model_gateway below: "litellm is present" and "the anthropic SDK is
# present" are orthogonal facts about a pipeline (see
# docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md), not competing answers.
DIRECT_SDK_MODULES = ["anthropic"]


def _module_is_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True)
class DetectedStack:
    orchestrators: frozenset[str] = field(default_factory=frozenset)
    model_gateway: str | None = None
    vector_db: str | None = None
    direct_sdks: frozenset[str] = field(default_factory=frozenset)


def detect_stack(is_available: Callable[[str], bool] = _module_is_available) -> DetectedStack:
    orchestrators = frozenset(m for m in ORCHESTRATOR_MODULES if is_available(m))
    model_gateway = "litellm" if is_available("litellm") else None
    vector_db = next((m for m in VECTOR_DB_MODULES if is_available(m)), None)
    direct_sdks = frozenset(m for m in DIRECT_SDK_MODULES if is_available(m))
    return DetectedStack(
        orchestrators=orchestrators, model_gateway=model_gateway, vector_db=vector_db, direct_sdks=direct_sdks
    )
