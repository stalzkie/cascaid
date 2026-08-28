"""E2E seam: `cascaid run -- <command>`, launched as a real subprocess -- proves
zero-code-change instrumentation actually works against a target script Cascaid
never imports directly, the way a real customer's app would be launched (Auto-
Instrumentation Glue Layer Plan, step 3)."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

# A bare litellm call with no LangGraph invocation around it has no current_step
# (see register_litellm_callbacks' no-op guard) and is correctly dropped -- this
# script wraps the call in a real, tiny LangGraph pipeline instead, the shape a
# real customer's app would actually take.
_TARGET_SCRIPT = """
import time
from typing import TypedDict

import litellm
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    query: str


def researcher(state, config):
    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["query"]}],
        mock_response="hello there",
    )
    return {}


g = StateGraph(State)
g.add_node("researcher", researcher)
g.add_edge(START, "researcher")
g.add_edge("researcher", END)
compiled = g.compile()
compiled.invoke({"query": "hi"})

time.sleep(0.5)  # give litellm's async success callback time to fire before exit
"""


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("timed out waiting for cascaid-run-launched subprocess to write events")


@pytest.mark.e2e
def test_run_instruments_a_real_subprocess_with_no_code_changes(tmp_path):
    script_path = tmp_path / "target_app.py"
    script_path.write_text(_TARGET_SCRIPT, encoding="utf-8")
    events_path = tmp_path / "events.jsonl"

    command = [
        sys.executable,
        "-m",
        "cascaid.cli",
        "run",
        "--events",
        str(events_path),
        "--",
        sys.executable,
        str(script_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    _wait_for(events_path.exists)

    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    topology_events = [line for line in lines if line["type"] == "topology"]
    assert len(topology_events) == 1
    assert topology_events[0]["nodes"] == {"researcher": "agent"}

    call_events = [line for line in lines if line["type"] == "call_event"]
    assert len(call_events) == 1
    assert call_events[0]["caller"] == "researcher"
    assert call_events[0]["callee"] == "gpt-4o-mini"
    assert call_events[0]["error"] is False
    assert call_events[0]["run_id"]  # a real run_id was generated and propagated into the child process
