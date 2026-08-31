"""Invoked automatically by a generated sitecustomize.py (prepended to PYTHONPATH
by `cascaid run`) before the target command's own code executes -- this is what
makes zero-code-change instrumentation possible for a real customer pipeline (PRD
4.1). Detects the stack (PRD 4.5) and wires only the adapters that apply; observed
topology/CallEvents are appended as JSON lines to CASCAID_EVENTS_PATH, reusing
CallEvent's existing to_json() rather than inventing a second serialization.
"""

from __future__ import annotations

import json
import os


def _file_sink(events_path: str):
    def sink(event) -> None:
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "call_event", **event.to_json()}) + "\n")

    return sink


def _topology_sink(events_path: str):
    def sink(nodes, edges) -> None:
        record = {
            "type": "topology",
            "nodes": {name: node_type.value for name, node_type in nodes.items()},
            "edges": [list(edge) for edge in edges],
        }
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return sink


def bootstrap() -> None:
    from cascaid.ingestion.runtime_context import current_run_id
    from cascaid.ingestion.stack_detector import detect_stack

    run_id = os.environ.get("CASCAID_RUN_ID")
    if run_id:
        current_run_id.set(run_id)

    events_path = os.environ.get("CASCAID_EVENTS_PATH")
    stack = detect_stack()

    if "langgraph" in stack.orchestrators:
        from cascaid.ingestion.langgraph_adapter import instrument_langgraph

        instrument_langgraph(topology_sink=_topology_sink(events_path) if events_path else (lambda n, e: None))
    if "crewai" in stack.orchestrators:
        from cascaid.ingestion.crewai_adapter import instrument_crewai

        instrument_crewai(topology_sink=_topology_sink(events_path) if events_path else (lambda n, e: None))

    if stack.model_gateway == "litellm":
        from cascaid.ingestion.litellm_adapter import register_litellm_callbacks

        register_litellm_callbacks(sink=_file_sink(events_path) if events_path else (lambda event: None))

    if "anthropic" in stack.direct_sdks:
        from cascaid.ingestion.anthropic_adapter import instrument_anthropic

        instrument_anthropic(sink=_file_sink(events_path) if events_path else (lambda event: None))

    if "openai" in stack.direct_sdks:
        from cascaid.ingestion.openai_adapter import instrument_openai

        instrument_openai(sink=_file_sink(events_path) if events_path else (lambda event: None))

    # pgvector is intentionally excluded: it's not a distinct client library (a
    # Postgres extension invoked through psycopg/SQLAlchemy), so reliably
    # detecting "this query is a vector similarity search" without false
    # positives needs more design than Pinecone/Weaviate's dedicated clients --
    # documented as a manual observe_vector_query() wrap for that stack.
    if stack.vector_db == "pinecone":
        from cascaid.ingestion.vector_query_adapter import register_pinecone_callbacks

        register_pinecone_callbacks(sink=_file_sink(events_path) if events_path else (lambda event: None))
    elif stack.vector_db == "weaviate":
        from cascaid.ingestion.vector_query_adapter import register_weaviate_callbacks

        register_weaviate_callbacks(sink=_file_sink(events_path) if events_path else (lambda event: None))
