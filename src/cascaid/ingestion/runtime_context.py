"""Propagates runtime-observation state across the ingestion seam (PRD section 4.5):
the LangGraph adapter sets these around each node's/invocation's execution, the
LiteLLM and vector-DB adapters read them when building a CallEvent.

current_run_id/current_step exist so `step` keeps meaning what it meant during
training -- one top-level pipeline invocation, not one raw call event (see the
Auto-Instrumentation Glue Layer Plan: fragmenting one invocation across several
call-indexed "steps" would be a train/serve distribution mismatch, not just an
integration shortcut). current_run_id has session lifetime (set once when
instrumentation starts); current_step is set fresh around each invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

current_node: ContextVar[str | None] = ContextVar("current_node", default=None)
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)
current_step: ContextVar[int | None] = ContextVar("current_step", default=None)


@contextmanager
def track_node(name: str):
    token = current_node.set(name)
    try:
        yield
    finally:
        current_node.reset(token)


@contextmanager
def track_run(run_id: str):
    token = current_run_id.set(run_id)
    try:
        yield
    finally:
        current_run_id.reset(token)


@contextmanager
def track_step(step: int):
    token = current_step.set(step)
    try:
        yield
    finally:
        current_step.reset(token)
