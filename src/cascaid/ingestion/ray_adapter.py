"""Distributed attribution across a Ray worker boundary (PRD 4.5's runtime seam,
extended past the single-process case runtime_context.py's contextvars cover on their
own). See docs/adr/0008-ray-distributed-attribution.md for why Ray next (after Celery,
ADR 0007) and how this design was arrived at.

Unlike every other adapter in this package (celery_adapter.py included), this produces no
CallEvents itself -- it only keeps current_run_id/current_step/current_node correct across
a real Ray worker process, so whichever adapter actually runs inside the task (litellm, a
direct SDK, ...) keeps attributing correctly.

Modeled on Ray's own built-in OpenTelemetry tracing integration
(ray/util/tracing/tracing_helper.py), which solves the identical problem: lazily wrap the
decorated function once (inside RemoteFunction._remote(), the same method that submits the
task -- Ray has no separate signal system the way Celery does) to accept a hidden
keyword-only param carrying context, and inject that param's value at every call site.

Two real footguns surfaced empirically before trusting this (see the ADR for the full
reproduction):

1. The wrapper's own closure must never reference a bare ContextVar object as a global --
   only the *module* (`from cascaid.ingestion import runtime_context as rc;
   rc.current_run_id.set(...)`). Cloudpickle serializes any closure (a function with
   captured/free variables) by value, inlining every global its bytecode touches; a raw
   ContextVar has no pickle support at all, so touching one directly breaks pickling for
   *every* wrapped task, not just ones that reference the ContextVar themselves. Modules
   and plain top-level functions pickle fine by reference (the worker just re-imports
   them).
2. `functools.wraps()` sets `__wrapped__`, and Ray's own argument-signature validation
   follows that back to the *original* function -- so the injected hidden kwarg gets
   rejected as "unexpected" unless the original function's `__signature__` is explicitly
   patched (mirroring Ray's own `_add_param_to_signature`) before wrapping.

Scope limit: covers `@ray.remote`-decorated functions only (`RemoteFunction`). Ray Actors
(`@ray.remote class Foo: ...`) are a materially different case Ray's own tracing
integration handles via a separate code path this adapter does not replicate.
"""

from __future__ import annotations

import functools
import inspect

from cascaid.ingestion import runtime_context as rc

_CTX_KWARG = "_cascaid_run_context"


def _wrap_function_for_context(function):
    old_sig = inspect.signature(function)
    if _CTX_KWARG not in old_sig.parameters:
        new_param = inspect.Parameter(_CTX_KWARG, inspect.Parameter.KEYWORD_ONLY, default=None)
        function.__signature__ = old_sig.replace(parameters=[*old_sig.parameters.values(), new_param])

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        ctx = kwargs.pop(_CTX_KWARG, None)
        if ctx is None:
            return function(*args, **kwargs)
        tokens = [
            (rc.current_run_id, rc.current_run_id.set(ctx.get("run_id"))),
            (rc.current_step, rc.current_step.set(ctx.get("step"))),
            (rc.current_node, rc.current_node.set(ctx.get("node"))),
        ]
        try:
            return function(*args, **kwargs)
        finally:
            for var, token in tokens:
                var.reset(token)

    return wrapped


def _make_patched_remote(original_remote):
    def patched(self, *args, **kwargs):
        if not getattr(self, "_cascaid_wrapped", False):
            self._function = _wrap_function_for_context(self._function)
            self._cascaid_wrapped = True
            # Force RemoteFunction._remote to re-derive _function_signature/re-pickle
            # from the now-wrapped self._function instead of a stale cached copy.
            self._function_signature = None

        run_id = rc.current_run_id.get()
        step = rc.current_step.get()
        node = rc.current_node.get()
        if run_id is not None or step is not None or node is not None:
            call_kwargs = dict(kwargs.get("kwargs") or {})
            call_kwargs[_CTX_KWARG] = {"run_id": run_id, "step": step, "node": node}
            kwargs["kwargs"] = call_kwargs

        return original_remote(self, *args, **kwargs)

    return patched


def instrument_ray() -> None:
    """Patches ray.remote_function.RemoteFunction._remote so run_id/step/node survive a
    task crossing into a separate Ray worker process. Guarded by
    __cascaid_instrumented__ so a repeated bootstrap call doesn't stack duplicate
    patches (same idempotency marker every monkeypatch-based adapter in this package
    uses).
    """
    from ray.remote_function import RemoteFunction

    if getattr(RemoteFunction._remote, "__cascaid_instrumented__", False):
        return

    patched = _make_patched_remote(RemoteFunction._remote)
    patched.__cascaid_instrumented__ = True
    RemoteFunction._remote = patched
