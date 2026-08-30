# Production Readiness & Pipeline Compatibility Assessment (2026-08-30)

Follow-up to `Client_Readiness_and_YC_Grade_Assessment.md`, narrowed to two
questions: what stands between today's Cascaid and a customer actually
running it in production, and how well does it handle pipelines that don't
look exactly like the demo. Same method as before -- every claim checked
against the current code, not the PRD's aspirational language.

## Headline finding: a confirmed, fixable bug in the most common pipeline shape

**Async LangGraph pipelines (`.ainvoke()`) get zero instrumentation, silently.**

`langgraph_adapter.py`'s `instrument_langgraph()` patches both
`compiled.invoke` and `compiled.ainvoke` with the same wrapper,
`_wrap_invoke()`:

```python
def _wrap_invoke(original_invoke, step_counter):
    def wrapped(*args, **kwargs):
        with track_step(next(step_counter)):
            return original_invoke(*args, **kwargs)

    return wrapped
```

For `invoke` (sync) this is correct. For `ainvoke`, `original_invoke(...)`
doesn't run the graph -- it *constructs a coroutine object* and returns
immediately, since `wrapped` itself is a plain `def`, not `async def`. The
`with track_step(...)` block's `__exit__` (which resets the
`current_step`/`current_run_id` contextvars) fires the instant that
coroutine is constructed -- before the caller ever `await`s it, i.e. before
the graph actually runs. By the time any node executes and calls out to
LiteLLM or a vector DB, `current_step` has already been reset, so
`register_litellm_callbacks`'/`register_pinecone_callbacks`'s own guard
(`if run_id is None or step is None: return`) makes every adapter silently
no-op. No error, no warning -- `cascaid run -- python my_async_app.py`
looks like it's working (the process runs normally) while producing an
empty event log for any graph invoked via `ainvoke`.

This isn't an edge case: `ainvoke` is the standard call for any
FastAPI-served or otherwise-async agent service, which is a large and
growing share of real LangGraph deployments -- arguably more common in
production than the sync `invoke` path the existing tests actually cover
(`grep -rn "ainvoke" tests/` returns nothing; this path has zero test
coverage, which is exactly how it shipped unnoticed).

**Fix**: `wrapped` needs to be `async def` when wrapping `ainvoke`, so the
`track_step` context manager stays entered for the coroutine's actual
execution, not just its construction -- two separate wrapper functions
(sync and async), not one shared one. `_wrap_node_fn` had the identical bug
for async node functions themselves (a plain sync wrapper around an async
node returns an unawaited coroutine, which LangGraph then rejects outright
with `InvalidUpdateError` -- worse than silent, but same root cause). Both
fixed the same way: `asyncio.iscoroutinefunction()` picks an `async def`
wrapper when needed.

**A second, deeper bug surfaced while fixing the first one and proving it
end-to-end**: even after both `ainvoke` wrappers were fixed, a real async
LangGraph pipeline calling `litellm.acompletion()` still produced zero
CallEvents. Verified empirically (small standalone repro scripts, not
guessed): litellm's legacy `success_callback`/`failure_callback` lists
(what `register_litellm_callbacks` used) never fire for `acompletion()` at
all -- only for sync `completion()`. litellm's modern mechanism for this,
a `CustomLogger` registered via `litellm.callbacks`, does cover both, but
further testing showed litellm defers the async half
(`async_log_success_event`) to its own internal background dispatch,
detached from the coroutine that made the call and often running well
after it returns -- by which point `current_run_id`/`current_step`/
`current_node` are already reset, so attribution would be wrong even if the
callback fired. Fixed by patching `litellm.completion`/`acompletion`
themselves to snapshot those contextvars into the call's own `metadata`
kwarg *at call time* (while still valid), and reading that snapshot back in
the async logger instead of the (by-then-stale) contextvars.

A first version of the fix moved sync dispatch onto `CustomLogger` too
(one mechanism for both), but this measurably slowed down litellm's own
callback dispatch under the full test suite's real load, enough to make
unrelated, previously-reliable tests' short timeouts flaky. Reverted to a
hybrid: sync `completion()` keeps using the legacy lists (proven fast,
unchanged from before), and only `acompletion()` uses the new
`CustomLogger` + metadata-snapshot mechanism. Both bugs (and this dispatch
mechanism split) are covered by deterministic unit tests that don't depend
on litellm's actual callback-firing timing -- see
`tests/unit/test_langgraph_adapter.py` and
`tests/unit/test_litellm_adapter.py`. I'm fixing all of this directly
rather than just flagging it, since it's a correctness bug in already-
shipped functionality, not a scope or design decision.

## Pipeline compatibility: what else breaks or is silently unsupported

Ranked by how many real pipelines each one likely affects:

1. **Direct OpenAI/Anthropic/Gemini SDK calls are invisible to Cascaid.**
   Only `litellm`'s callback registry is patched
   (`register_litellm_callbacks`). A pipeline calling
   `openai.chat.completions.create(...)` or `anthropic.messages.create(...)`
   directly -- arguably *more* common than routing every call through
   LiteLLM as a gateway -- gets no model-endpoint observability at all, with
   no error or indication that anything is missing. This is probably the
   single largest "will this work on my pipeline" gap, larger than the
   orchestrator-framework question below, because it affects LangGraph and
   CrewAI pipelines alike regardless of which orchestrator is used.
2. **CrewAI's `kickoff_async` isn't patched** (`crewai_adapter.py`'s own
   docstring names this as a known simplification) -- unlike the LangGraph
   bug above, this at least fails safe: the original unpatched method runs
   untouched, so there's no false confidence, just no instrumentation for
   async CrewAI users.
3. **AutoGen and hand-rolled/custom orchestration remain unsupported**
   (reaffirming the prior assessment's finding). No generic, manual
   instrumentation seam is documented as a fallback -- e.g. a simple
   `with cascaid.track_node("my_step"):` a customer could drop into code
   Cascaid doesn't auto-detect. Today the only fallback for an unsupported
   stack is silence, not a documented manual path.
4. **Vector DB coverage stops at Pinecone and Weaviate** (auto-patched) and
   pgvector (manual one-liner, by design). Chroma, Qdrant, Milvus, and
   LanceDB -- all in real production use -- aren't supported at all.
   Checking the two that *are* supported precisely (not trusting the
   "verified via introspection" comment as still true) found two more real
   bugs, both fixed: pinecone's pinned `9.1.0` added `Index.search_records`
   (an alias for `search`) and `Index.fetch_by_metadata` since
   `PINECONE_QUERY_METHODS` was last checked -- silently under-counted,
   same failure class as everything else in this list. And weaviate-client
   ships a fully separate async client (`WeaviateAsyncClient`/
   `_QueryCollectionAsync`) with identical method names to the sync
   `_QueryCollection` this module already patched -- a customer using
   Weaviate's own recommended async client got zero instrumentation, and a
   naive fix would have hit the exact sync-wrapper-around-async-method bug
   already fixed twice this session (LangGraph, CrewAI) -- caught before
   shipping it a third time.
5. **Contextvar-based attribution is process-local.** `current_run_id`/
   `current_step`/`current_node` are `contextvars.ContextVar`s, which don't
   cross process boundaries and need explicit propagation across
   `asyncio` tasks spawned without inheriting context, or across a thread
   pool. A pipeline that fans work out across worker processes (Celery,
   Ray, `multiprocessing`) would lose attribution partway through. Fine for
   a v1 single-process target; a real ceiling on "the entire client base"
   once pipelines get more distributed.
6. **The model's generalization to a real, differently-shaped topology is
   still unvalidated** -- carried over from the prior assessment and from
   `GNN_Accuracy_Improvement_Log.md`'s own "what's still open" section. The
   architecture is genuinely inductive (verified in the last assessment),
   but every fault scenario the 0.90 PR-AUC number is based on still comes
   from the same 7-node synthetic graph. This remains the single largest
   unknown behind any accuracy claim on a customer's actual pipeline, and
   is worth restating because it's more fundamental than any item above.

## Production readiness: what's missing beyond a working demo

1. **No database migrations.** `init_db()` is `Base.metadata.create_all(engine)`
   only -- no Alembic. `create_all` creates missing tables but never alters
   an existing one. Harmless so far (nothing in this session's work touched
   the SQLAlchemy schema), but the moment a future release needs to add or
   change a column on `score_history`/`incident_labels`/etc., an existing
   customer's Postgres won't pick it up automatically on upgrade -- this is
   a schema decision worth making before it's needed under pressure, not
   after a customer's upgrade silently breaks.
2. **No data retention policy, despite a code comment implying one exists.**
   `storage/models.py`'s own docstring says "TimescaleDB hypertables on
   score_history/alert_history are an ops-level upgrade -- see
   docker/postgres/init.sql" -- that file does not exist anywhere in the
   repo. `score_history` gets a row per node on every risk check, and the
   dashboard auto-refreshes every 10 seconds per open run -- in real
   continuous production use this table has no partitioning, retention, or
   archival story and grows genuinely unbounded.
3. **No restart policy or health checks on `serve`/`dashboard` in
   docker-compose.** Only the `postgres` service has a healthcheck; if the
   model-serving or dashboard process crashes in production, Compose won't
   restart it (`restart: unless-stopped` is absent everywhere).
4. **No model-config/version metadata.** A saved model file records only
   its weights, not the hyperparameters (`hidden`, `layers`, `conv`) it was
   trained with. `cascaid serve`/`cascaid retrain` currently agree by
   accident (both default to the same `CascadeGNN` defaults, and neither
   exposes `--layers`/`--conv` on its CLI) -- the moment that changes, a
   mismatched load fails with a low-level PyTorch state-dict error instead
   of a clear "this model was trained with different hyperparameters"
   message.
5. **CORS wildcard on the dashboard API** (`allow_origins=["*"]` in
   `dashboard/api.py`) -- fine behind a token-auth API for a local demo, a
   hardening item before this sits behind a real hostname other pages could
   embed requests from.
6. **Secrets stored in plaintext** in the `Config` key-value table
   (`llm_api_key`, `alert_pagerduty_routing_key`, alongside the existing
   webhook URL) -- consistent with how the codebase already stores this
   class of value, but worth naming together as "everything here is
   plaintext at rest," a real question for security-conscious buyers given
   the PRD's own enterprise/VPC pitch.
7. **No backup/restore documentation** for the self-hosted Postgres data --
   nothing describes how a customer would back up or restore incident
   history and score history.

## What I'm doing now vs. flagging for a decision

**Fixing now** (bug fix in existing code, no design decision, no public API
change): the async LangGraph `ainvoke` instrumentation bug above.

**Flagging, not building yet** -- each of these is either a genuine
infra/architecture decision (worth a quick confirmation before I sink time
into it) or a larger net-new build than a same-session bug fix:
- Introducing Alembic migrations (a real workflow change for every future
  schema edit, not just this one)
- Designing and shipping an actual retention policy (TimescaleDB
  hypertables vs. a simpler periodic-delete job vs. just documenting a
  customer-side cron -- genuine trade-off, not obvious)
- Direct OpenAI/Anthropic/Gemini SDK instrumentation (a new adapter, same
  shape as the existing LiteLLM one, but real new-build effort)
- A generic manual-tracking fallback SDK for unsupported frameworks
- Additional vector DB adapters (Chroma/Qdrant/Milvus/LanceDB)
- `docker-compose.yml` restart policies + serve/dashboard healthchecks
  (small, but changes how the stack behaves operationally, worth a nod
  before changing)
