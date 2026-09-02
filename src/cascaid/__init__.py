"""Cascaid's customer-facing public API: manual instrumentation for frameworks Cascaid
doesn't auto-detect (the `ag2` fork of AutoGen, hand-rolled orchestration, an in-house
model client). See docs/adr/0002-manual-tracking-sdk-is-context-manager-shaped.md.

Everything auto-detected (LangGraph, CrewAI, AutoGen via `autogen-agentchat` -- see
docs/adr/0006-autogen-agentchat-not-ag2.md, litellm, direct Anthropic/OpenAI/Gemini SDKs,
vector DBs) is wired up automatically by `cascaid run` -- nothing in this module is
needed for those. This is only for the gap: a pipeline shape Cascaid can't see into on
its own.

Usage (hand-rolled orchestration, e.g. a raw custom agent loop)::

    import cascaid

    with cascaid.track_run("run-1"):
        for step, task in enumerate(my_tasks):
            with cascaid.track_step(step):
                with cascaid.observe_call("my_model_endpoint", cascaid.NodeType.MODEL_ENDPOINT) as call:
                    response = my_custom_llm_client.generate(task)
"""

from __future__ import annotations

from cascaid.ingestion.manual_adapter import observe_call, observe_call_async
from cascaid.ingestion.runtime_context import track_run, track_step
from cascaid.ingestion.schema import NodeType

__all__ = ["NodeType", "observe_call", "observe_call_async", "track_run", "track_step"]
