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

    if stack.orchestrator == "langgraph":
        from cascaid.ingestion.langgraph_adapter import instrument_langgraph

        instrument_langgraph(topology_sink=_topology_sink(events_path) if events_path else (lambda n, e: None))

    if stack.model_gateway == "litellm":
        from cascaid.ingestion.litellm_adapter import register_litellm_callbacks

        register_litellm_callbacks(sink=_file_sink(events_path) if events_path else (lambda event: None))
