"""Distributed attribution across a Celery worker boundary (PRD 4.5's runtime seam,
extended past the single-process case runtime_context.py's contextvars cover on their
own). See docs/adr/0007-celery-first-for-distributed-attribution.md for why Celery first
and why this propagates via task headers + signals rather than a customer-facing API.

Unlike every other adapter in this package, this one produces no CallEvents itself --
current_run_id/current_step/current_node are plain contextvars, so they don't survive a
task crossing into a separate worker process on their own. This only keeps them correct
across that boundary, so whichever adapter actually runs inside the task body (litellm, a
direct SDK, LangGraph, ...) keeps attributing correctly.

Verified empirically before writing this (see the ADR): `before_task_publish` is the one
seam every dispatch path (`.delay()`, `.apply_async()`, canvas primitives) funnels
through, and its `headers` dict is genuinely mutable on the outgoing message. `task_prerun`
exposes those same headers worker-side via `task.request.headers`; `task_postrun` fires
even when the task raised, so the pair gives a reliable set/reset boundary around the task
body. Signal.connect() defaults to weak=True, which would let these closures get
garbage-collected the moment this function returns (confirmed: the handler silently stops
firing) -- weak=False is required on every connect() call here.

Scope limit: verified for prefork/solo/thread-based worker pools, where each task fully
owns one OS thread for its duration. Greenlet-based pools (eventlet/gevent) were not
verified -- contextvars isolation across greenlets sharing one OS thread depends on the
greenlet library's own context-copying support, not something this adapter controls.
"""

from __future__ import annotations

from celery import signals

from cascaid.ingestion.runtime_context import current_node, current_run_id, current_step

_HEADER_RUN_ID = "cascaid_run_id"
_HEADER_STEP = "cascaid_step"
_HEADER_NODE = "cascaid_node"

_VAR_BY_HEADER = {
    _HEADER_RUN_ID: current_run_id,
    _HEADER_STEP: current_step,
    _HEADER_NODE: current_node,
}

# task_id -> list of (ContextVar, Token) pairs set by _on_prerun, reset by _on_postrun.
# Keyed by task_id (not a stack) since concurrent tasks in a threaded pool each get their
# own entry; a solo/prefork pool only ever has one live entry at a time.
_active_tokens: dict[str, list] = {}


def _on_before_task_publish(sender=None, headers=None, **kwargs) -> None:
    if headers is None:
        return
    run_id = current_run_id.get()
    if run_id is not None:
        headers[_HEADER_RUN_ID] = run_id
    step = current_step.get()
    if step is not None:
        headers[_HEADER_STEP] = step
    node = current_node.get()
    if node is not None:
        headers[_HEADER_NODE] = node


def _on_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **rest) -> None:
    headers = getattr(getattr(task, "request", None), "headers", None) or {}
    tokens = []
    for header_key, var in _VAR_BY_HEADER.items():
        if header_key in headers:
            tokens.append((var, var.set(headers[header_key])))
    if tokens:
        _active_tokens[task_id] = tokens


def _on_task_postrun(sender=None, task_id=None, task=None, retval=None, state=None, **rest) -> None:
    tokens = _active_tokens.pop(task_id, None)
    if not tokens:
        return
    for var, token in tokens:
        var.reset(token)


def instrument_celery() -> None:
    """Connects to celery.signals.before_task_publish/task_prerun/task_postrun so
    run_id/step/node survive a task crossing into a separate Celery worker. dispatch_uid
    makes each connect() idempotent -- a repeated bootstrap call doesn't stack duplicate
    handlers (the signal-based equivalent of the __cascaid_instrumented__ marker every
    monkeypatch-based adapter in this package uses for the same purpose). weak=False is
    required: these are closures with no other strong reference, and Signal.connect()
    defaults to weak=True, which would let them get garbage-collected right after this
    function returns.
    """
    signals.before_task_publish.connect(
        _on_before_task_publish, dispatch_uid="cascaid_celery_before_task_publish", weak=False
    )
    signals.task_prerun.connect(_on_task_prerun, dispatch_uid="cascaid_celery_task_prerun", weak=False)
    signals.task_postrun.connect(_on_task_postrun, dispatch_uid="cascaid_celery_task_postrun", weak=False)
