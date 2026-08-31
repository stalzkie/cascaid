"""Smoke test for the customer-facing `import cascaid` namespace (see
cascaid/__init__.py and docs/adr/0002-manual-tracking-sdk-is-context-manager-shaped.md).
Behavior is already covered by test_manual_adapter.py and test_runtime_context.py (if
present) -- this only proves the public re-exports exist and are the real objects, not
placeholders."""

import cascaid
from cascaid.ingestion.manual_adapter import observe_call, observe_call_async
from cascaid.ingestion.runtime_context import track_run, track_step
from cascaid.ingestion.schema import NodeType


def test_public_namespace_exposes_observe_call():
    assert cascaid.observe_call is observe_call


def test_public_namespace_exposes_observe_call_async():
    assert cascaid.observe_call_async is observe_call_async


def test_public_namespace_exposes_track_run_and_track_step():
    assert cascaid.track_run is track_run
    assert cascaid.track_step is track_step


def test_public_namespace_exposes_node_type():
    assert cascaid.NodeType is NodeType
