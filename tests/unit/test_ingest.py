"""Unit seam: cascaid.ingest.read_events_file's JSON-lines parsing."""

from __future__ import annotations

import json

from cascaid.ingest import read_events_file
from cascaid.ingestion.schema import NodeType


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_read_events_file_returns_empty_when_file_does_not_exist(tmp_path):
    nodes, edges, events = read_events_file(tmp_path / "does-not-exist.jsonl")
    assert nodes == {}
    assert edges == []
    assert events == []


def test_read_events_file_parses_topology_and_call_events(tmp_path):
    path = tmp_path / "events.jsonl"
    _write(
        path,
        [
            {"type": "topology", "nodes": {"researcher": "agent"}, "edges": []},
            {
                "type": "call_event",
                "run_id": "run-1",
                "scenario": "production",
                "step": 0,
                "caller": "researcher",
                "callee": "gpt-4o-mini",
                "caller_type": "agent",
                "callee_type": "model_endpoint",
                "latency_ms": 12.5,
                "error": False,
                "retried": False,
                "token_cost": 0.001,
            },
        ],
    )

    nodes, edges, events = read_events_file(path)

    assert nodes == {"researcher": NodeType.AGENT}
    assert edges == []
    assert len(events) == 1
    assert events[0].caller == "researcher"
    assert events[0].callee == "gpt-4o-mini"


def test_read_events_file_uses_the_last_topology_record_when_multiple_exist(tmp_path):
    path = tmp_path / "events.jsonl"
    _write(
        path,
        [
            {"type": "topology", "nodes": {"old_node": "agent"}, "edges": []},
            {"type": "topology", "nodes": {"new_node": "tool"}, "edges": []},
        ],
    )

    nodes, edges, events = read_events_file(path)

    assert nodes == {"new_node": NodeType.TOOL}
