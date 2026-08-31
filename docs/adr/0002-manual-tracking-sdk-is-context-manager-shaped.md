---
status: accepted
---

# The manual-tracking fallback SDK (for frameworks Cascaid doesn't auto-detect) is context-manager shaped

Cascaid's auto-instrumentation covers what `stack_detector.py` can detect (LangGraph,
CrewAI, litellm, direct Anthropic/OpenAI SDKs, Pinecone/Weaviate/...). For anything it
doesn't -- AutoGen, hand-rolled orchestration, an in-house model client -- a customer
needs a manual way to tell Cascaid "this was a call." That manual API is a new public
surface, worth deciding deliberately rather than defaulting to whatever's easiest to
implement: it's the one piece of this project a customer writes code against directly.

We're shipping it as a context manager -- `with cascaid.observe_call(callee=...,
callee_type=...) as call: ...` -- matching the pattern already used everywhere in this
codebase (`runtime_context.track_node`/`track_run`/`track_step`,
`vector_query_adapter.observe_vector_query`/`observe_vector_query_async`). Cascaid times
the block and builds the `CallEvent` automatically, including catching an exception
raised inside the block as `error=True` before re-raising it -- the customer never
computes latency or wires up their own try/except for attribution.

## Considered Options

- **Decorator** (`@cascaid.trace(callee=...)`, rejected): reads well for a call
  isolated to its own function, but a customer's manual-tracking need is usually an
  inline call inside a larger method they don't want to extract just to get a
  decorator target, and inferring error state from a wrapped function's return value
  needs more guessing than a context manager's explicit exception propagation.
- **Explicit function call** (`cascaid.record_call(caller=..., callee=...,
  latency_ms=..., error=..., ...)`, rejected): maximum flexibility, least code for
  Cascaid to write, but pushes all the timing and exception bookkeeping onto the
  customer -- the wrong trade for a product whose whole premise (see
  `_instrument_bootstrap.py`'s module docstring) is zero-code-change, low-effort
  instrumentation everywhere else.

## Consequences

- `run_id`/`step`/`caller` are still read from `runtime_context`'s contextvars, same as
  every other adapter -- `observe_call` is a manual instrumentation *point*, not a
  parallel attribution mechanism.
