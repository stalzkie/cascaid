---
status: accepted
---

# Ray next for distributed attribution, propagated via a wrapped-function hidden kwarg

Follow-up to ADR 0007 (Celery). `docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md`'s
gap #3 named Ray and bare `multiprocessing` as deferred, not scoped. This scopes and ships
Ray; `multiprocessing` remains deferred.

## Which backend: Ray over bare `multiprocessing`

Ray has real prior art to model the design after: its own built-in OpenTelemetry tracing
integration (`ray/util/tracing/tracing_helper.py`, installed alongside `ray` itself) solves
exactly this problem -- propagating ambient context across a real Ray worker process
boundary -- using a pattern this ADR follows closely. Bare `multiprocessing` has no
comparable prior art in the stdlib or a maintained library to draw from, and (confirmed
empirically) only supports the `spawn` start method on Windows, meaning zero implicit
memory sharing across the boundary under any circumstance -- higher first-pass risk with
no reference implementation to check against. Ray is also closer to Cascaid's actual space
than originally assumed in ADR 0007 (LLM/agent batch workloads increasingly run on Ray),
not purely an ML-cluster-scale concern.

Verified Ray actually runs on this Windows dev machine before investing further (`ray.init()`
+ a real `.remote()` task, confirmed working) -- not assumed from the package installing.

## How it works: verified empirically, two real footguns caught before trusting the design

Ray's tracing integration wraps the decorated function once (lazily, inside
`RemoteFunction._remote()`, the same method that submits the task) to accept a hidden
keyword-only param carrying serialized context, and injects that param's value at every
call site. Cascaid's `ray_adapter.py` follows the identical two-effects-in-one-method
shape: patch `RemoteFunction._remote()` itself (no separate signal system exists for Ray,
unlike Celery) to (a) lazily wrap `self._function` exactly once per `RemoteFunction`
instance, and (b) inject the current `run_id`/`step`/`node` into the call's kwargs on
every invocation.

A naive first attempt, modeled too literally on the wrap shape without checking what the
wrapper's own closure references, failed twice before it worked:

1. **Wrapping *any* Ray task breaks if the wrapper's own closure captures a bare
   `ContextVar` object as a global.** Cloudpickle serializes a closure (any function with
   captured/free variables) *by value*, which means it inlines every global name the
   closure's bytecode touches -- and `contextvars.ContextVar` has no pickle support at
   all. Confirmed by reproduction: a completely unrelated task (`def add(a, b): return a +
   b`, no reference to any ContextVar in its own body) failed to pickle the moment it got
   wrapped by a closure whose *own* body wrote `current_run_id.set(...)` -- because
   `current_run_id` the bare object, not `runtime_context` the module, was the global the
   wrapper's bytecode referenced. Fixed by having the wrapper reference the *module*
   (`from cascaid.ingestion import runtime_context as rc; rc.current_run_id.set(...)`)
   instead of importing the bare `ContextVar` name into its own scope -- modules and plain
   top-level functions pickle fine by reference (the worker just re-imports them); only
   the raw stateful object doesn't. This is exactly why Ray's own tracing wrapper only
   ever touches `_DictPropagator`/`tracer` (importable, stateless-to-pickle) and never a
   raw OpenTelemetry context object directly.
2. **`functools.wraps()` sets `__wrapped__`, and ray's own argument-signature validation
   follows it back to the original function** -- so the injected hidden kwarg gets
   rejected as "unexpected keyword argument" unless the *original* function's
   `__signature__` is explicitly patched to declare the new keyword-only parameter before
   wrapping (confirmed via reproduction: adding the parameter directly on `wrapped`'s own
   signature was not enough). Fixed by mirroring Ray's own
   `_add_param_to_signature`/`_inject_tracing_into_function` exactly: mutate
   `function.__signature__` on the *original* function object first, then wrap.

Confirmed against a real cross-process worker (`ray.init()` + `.remote()`, not eager/local
execution) after both fixes: a plain unrelated task pickles and runs correctly, a task that
reads `current_run_id` from a code path unconnected to its own arguments (simulating an
adapter running inside the task, e.g. litellm) correctly sees the driver's value across
the process boundary, and a task dispatched with no context set on the driver correctly
sees `None` worker-side -- no false propagation.

## Consequences

- `stack_detector.py` gains `"ray": "ray"` in `DISTRIBUTED_BACKEND_MODULES` (alongside
  `"celery": "celery"`), detected independently like every other entry.
- `ray_adapter.py`'s `instrument_ray()` produces no `CallEvent`s itself, same shape as
  `celery_adapter.py` -- it only keeps `current_run_id`/`current_step`/`current_node`
  correct across the worker boundary for whichever adapter actually runs inside the task.
- **Scope limit, stated plainly:** covers `@ray.remote`-decorated *functions*
  (`ray.remote_function.RemoteFunction`) only. Ray Actors (`@ray.remote class Foo: ...`,
  `ray.actor.ActorClass`) are a materially different case -- persistent stateful workers
  dispatching methods rather than one-shot task submission -- and Ray's own tracing
  integration handles them via a separate code path
  (`_inject_tracing_into_class`/`_tracing_actor_creation`) this adapter does not replicate.
  Documented simplification, not a correctness bug, same category as ADR 0007's
  unverified greenlet pools.
- A task dispatched with no `run_id`/`step`/`node` set runs with the same defaults as
  today -- no regression for pipelines that don't set attribution context before
  dispatching Ray tasks at all.
- Bare `multiprocessing` remains deferred (ADR 0007's original reasoning stands: no
  message-envelope or decorator seam to lean on, `spawn`-only on Windows means zero
  implicit sharing) -- revisit if a customer actually needs it.
- **Not verified in this pass, stated plainly rather than assumed:** whether a real Ray
  worker process spawned by `cascaid run` actually re-runs cascaid's own bootstrap
  (`_instrument_bootstrap.py`, normally triggered via a generated `sitecustomize.py`
  prepended to `PYTHONPATH`). This adapter keeps `current_run_id`/`current_step`/
  `current_node` correct *once a worker process has cascaid's other adapters patched in
  it* -- if Ray's worker startup doesn't inherit/re-apply that `PYTHONPATH` prepend the
  same way a plain subprocess does, litellm/direct-SDK calls inside the task would run
  unpatched regardless of whether the context values themselves arrived correctly. This
  session's tests instrument the test process directly (`instrument_ray()` called inline,
  same pattern `test_celery_adapter.py`/other adapter tests use); an end-to-end
  `cascaid run` against a real multi-process Ray cluster hasn't been run.
