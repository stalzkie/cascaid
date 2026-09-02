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

# Maps the vendor label used in DetectedStack.orchestrators/_instrument_bootstrap.py to
# the actual importable module, same reasoning as VECTOR_DB_MODULES/DIRECT_SDK_MODULES
# below: langgraph/crewai's import name happens to coincide with their label, but
# autogen-agentchat's does not -- it imports as `autogen_agentchat`, not `autogen` (which
# is the community `ag2` fork's import name instead, a different package -- see
# docs/adr/0006-autogen-agentchat-not-ag2.md).
ORCHESTRATOR_MODULES = {"langgraph": "langgraph", "crewai": "crewai", "autogen": "autogen_agentchat"}
# Vector DBs, detected independently like ORCHESTRATOR_MODULES/DIRECT_SDK_MODULES, not
# exclusively via next() as this used to be: a pipeline can genuinely use two vector DBs
# at once (e.g. Pinecone in prod, Chroma for local dev/testing in the same codebase) --
# exclusive detection would silently under-instrument that, the same class of accuracy
# problem PINECONE_QUERY_METHODS' own re-verification note in vector_query_adapter.py
# already worries about. Maps the vendor label used in DetectedStack.vector_dbs/
# _instrument_bootstrap.py to the actual importable module -- qdrant-client's package
# imports as `qdrant_client`, chromadb's vendor label "chroma" imports as `chromadb`;
# neither coincides with its label the way pinecone/weaviate/pgvector do.
VECTOR_DB_MODULES = {
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "pgvector": "pgvector",
    "chroma": "chromadb",
    "qdrant": "qdrant_client",
    "milvus": "pymilvus",
    "lancedb": "lancedb",
}
# Direct model-provider SDKs, detected independently like ORCHESTRATOR_MODULES, not
# exclusively like model_gateway below: "litellm is present" and "the anthropic SDK is
# present" are orthogonal facts about a pipeline (see
# docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md), not competing answers.
# Maps the vendor label used in DetectedStack.direct_sdks/_instrument_bootstrap.py to
# the actual dotted module `importlib` needs to probe -- these coincide for
# anthropic/openai, but google-genai's importable package is `google.genai`, not
# literally "gemini" (verified via introspection, see gemini_adapter.py).
DIRECT_SDK_MODULES = {"anthropic": "anthropic", "openai": "openai", "gemini": "google.genai"}


def _module_is_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True)
class DetectedStack:
    orchestrators: frozenset[str] = field(default_factory=frozenset)
    model_gateway: str | None = None
    vector_dbs: frozenset[str] = field(default_factory=frozenset)
    direct_sdks: frozenset[str] = field(default_factory=frozenset)


def detect_stack(is_available: Callable[[str], bool] = _module_is_available) -> DetectedStack:
    orchestrators = frozenset(label for label, module in ORCHESTRATOR_MODULES.items() if is_available(module))
    model_gateway = "litellm" if is_available("litellm") else None
    vector_dbs = frozenset(label for label, module in VECTOR_DB_MODULES.items() if is_available(module))
    direct_sdks = frozenset(label for label, module in DIRECT_SDK_MODULES.items() if is_available(module))
    return DetectedStack(
        orchestrators=orchestrators, model_gateway=model_gateway, vector_dbs=vector_dbs, direct_sdks=direct_sdks
    )
