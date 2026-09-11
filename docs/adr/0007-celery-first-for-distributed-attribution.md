---
status: accepted
---

# Celery first for distributed attribution, propagated via task headers + signals

`runtime_context.py`'s `current_run_id`/`current_step`/`current_node` are plain
`contextvars.ContextVar`s (PRD 4.5's attribution mechanism). They propagate correctly
across `asyncio` tasks within one process (confirmed for AutoGen, ADR 0006) but not across
a real distributed-worker boundary: a pipeline that fans work out to Celery loses
attribution the moment a task crosses into a separate worker process. Flagged as a
scoping decision in `docs/Production_Readiness_and_Pipeline_Compatibility_Assessment.md`
(#3), not started speculatively — same reasoning the direct-SDK adapters used before
being built (ADR 0001).

Two questions needed answering before writing any code: which backend to support first,
and how a context should propagate across an arbitrary worker boundary given there's no
`asyncio.copy_context()`-equivalent that works generically for that.

## Which backend first: Celery

Cascaid's actual customers are LangGraph/CrewAI/AutoGen pipelines deployed behind
task queues in ordinary web-shaped stacks, not ML-scale distributed compute clusters —
Celery is the standard choice for that shape of deployment and is what a real customer
pipeline is most likely to already have in front of an agent pipeline. Ray targets a
different deployment shape (distributed training/serving at cluster scale) and is a much
less likely fit for the orchestrators Cascaid already instruments. Bare `multiprocessing`
has no task-queue semantics or message envelope to lean on, making clean propagation
harder for a shape of usage that's also less likely to be what a real pipeline uses
directly. Ray/`multiprocessing` support is deferred, not ruled out — revisit if a customer
actually needs one.

## How to propagate: auto-serialize into the task payload, not a customer-facing API

Considered:

- **Explicit customer-facing API** (`propagate_context()` at the worker boundary,
  customer wires it themselves): simpler to implement and honest about the real limit
  (no universal auto-propagation across a process boundary), but breaks the "zero code
  changes" promise (PRD 4.1) that every other adapter in this codebase has kept so far —
  this would be the first one to ask the customer to change their own code.
- **Auto-serialize into the task payload** (chosen): keep the zero-code-change property
  by hooking Celery's own message-dispatch seam instead of any particular call site.

## What the real API looks like (verified empirically, not assumed)

Two footguns surfaced before anything was trusted enough to build on:

1. **`task_always_eager=True` bypasses the publish step entirely.** A first repro under
   eager mode never fired `before_task_publish` at all, yet `current_run_id` still read
   correctly *inside* the task body — because eager mode runs the task inline in the
   calling process/thread, so contextvars survive trivially with no real boundary crossed.
   That result would have looked like propagation working when it was actually testing
   nothing. Re-verified against a real boundary instead: `filesystem://` broker transport
   (real files as queues, no Redis/RabbitMQ needed) plus an actual `celery worker`
   subprocess consuming from it. Confirmed cross-process: publisher and worker logged
   different PIDs, and `task.request.headers` on the worker side correctly recovered what
   the publisher's process had written.
2. **Signal receivers are weakly referenced by default.** `Signal.connect(fn, ...)`
   defaults to `weak=True`; a closure defined inside `instrument_celery()` with no other
   strong reference gets garbage-collected the moment the function returns, and the
   handler silently stops firing (confirmed with a minimal repro: `gc.collect()` after
   connecting drops the handler, `sig.send()` afterward calls nothing). `weak=False` is
   required on every `connect()` call here.

Confirmed via the same repro:

- `before_task_publish` is the one seam every dispatch path funnels through — `.delay()`,
  `.apply_async()`, and canvas primitives (`chain`/`group`/`chord`) all call into it, same
  "one shared seam regardless of entry point" property `ChatAgentContainer.handle_request`
  had for AutoGen (ADR 0006). It hands a mutable `headers` dict on the outgoing message —
  writing `run_id`/`step`/`node` into it there requires no per-call-site patching.
- `task_prerun` (worker side, before the task body runs) exposes those same headers via
  `task.request.headers`, `task_postrun` fires *even when the task raised* (confirmed:
  the failure-path repro still logged `task_postrun` with `state=FAILURE`) — the pair
  is a reliable set/reset boundary around the task body, mirroring
  `track_run`/`track_step`/`track_node`'s own `contextvar.set()` + `try/finally reset()`
  shape.
- `dispatch_uid` on `connect()` dedups repeated `connect()` calls for the same identifier
  (confirmed: a second `connect()` with an already-used `dispatch_uid` is a no-op) — this
  is the idempotency guarantee a repeated `instrument_celery()` bootstrap call needs, the
  signal-based equivalent of the `__cascaid_instrumented__` marker every monkeypatch-based
  adapter uses for the same purpose.

## Consequences

- `stack_detector.py` gains a `distributed_backends: frozenset[str]` field on
  `DetectedStack`, detected independently like `orchestrators`/`direct_sdks`
  (`DISTRIBUTED_BACKEND_MODULES = {"celery": "celery"}`) — a new category, not folded into
  an existing one, since Celery is neither an orchestrator nor a model gateway.
- `celery_adapter.py`'s `instrument_celery()` takes no `sink` — unlike every other
  adapter, it doesn't produce `CallEvent`s itself; it only keeps
  `current_run_id`/`current_step`/`current_node` correct across the worker boundary so
  whichever adapters run *inside* the task body (litellm, a direct SDK, LangGraph, …)
  keep attributing correctly. `_instrument_bootstrap.py` wires it unconditionally when
  `"celery" in stack.distributed_backends`.
- **Scope limit, stated plainly, not silently assumed:** verified correct for prefork,
  `solo`, and thread-based worker pools, where each task fully owns one OS thread for its
  duration (contextvars are thread-local, and `task_prerun`/`task_postrun` run in that same
  thread). Greenlet-based pools (`eventlet`/`gevent`) were not verified — `contextvars`
  isolation across greenlets in the same OS thread depends on the greenlet library's own
  context-copying support, not something this adapter controls or checked. Documented
  limitation, not a blocker, same category as ADR 0006's un-recursed nested `Team`
  participant.
- A task published with no `run_id`/`step`/`node` set (headers carry none of the
  `cascaid_*` keys) runs with the same defaults as today — no regression for pipelines
  that don't set attribution context before dispatching Celery tasks at all.
