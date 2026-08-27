"""Propagates the currently-executing pipeline node name across the runtime-observation
seam (PRD section 4.5): the LangGraph adapter sets it around each node's execution, the
LiteLLM callback adapter reads it as the caller of a model-endpoint call.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

current_node: ContextVar[str | None] = ContextVar("current_node", default=None)


@contextmanager
def track_node(name: str):
    token = current_node.set(name)
    try:
        yield
    finally:
        current_node.reset(token)
