# Production Readiness & Pipeline Compatibility Assessment (2026-08-30)

## Session status (2026-08-31) — pausing here, continuing next session

**Compatibility pass: done for this round.** Five real bugs found and fixed
(all merged to `staging`, PRs #44-#47 + a docs-only precision PR #48) --
see "Pipeline compatibility" below for the full list (LangGraph `ainvoke`,
CrewAI `async_execution`, Pinecone method drift, Weaviate's async client,
LiteLLM streaming). All five shared one root cause: an implicit thread/task
boundary Cascaid's contextvar-based attribution didn't account for. Each
has a regression test; `staging` is green (248 tests, run 2x consecutively
for stability before merging each one).

**Update (2026-08-31, later same window):** the scoping conversation happened
and direct Anthropic SDK instrumentation shipped -- PR #50, merged to
`staging`. See `docs/adr/0001-anthropic-before-openai-direct-sdk-adapter.md`
for the decision record: Anthropic first (not OpenAI) because litellm's
OpenAI provider internally uses the real `openai.OpenAI`/`AsyncOpenAI`
client (confirmed in `litellm/llms/openai/openai.py`), so a direct OpenAI
adapter would double-count calls litellm itself already sinks -- litellm's
Anthropic path has no such dependency. `DetectedStack` gained
`direct_sdks: frozenset[str]`, detected independently of `model_gateway`
(same pattern as `orchestrators`). 12 new tests, full suite green twice
(258 passed, 2 skipped both runs).

**Not started yet, pick up next session:**
- Direct OpenAI SDK instrumentation -- the "coexist with dedup" design is
  decided (see the ADR above) but not built: the OpenAI adapter's patched
  `create()` needs to check for cascaid's own litellm metadata marker on a
  call and skip it if litellm's adapter already sinked a `CallEvent` for
  it, so litellm + a raw `openai.Client()` in the same pipeline compose
  instead of double-counting.
- Direct Gemini SDK instrumentation -- not scoped yet at all (unlike
  OpenAI, no double-counting risk has been checked; litellm's Gemini/Vertex
  path was only glanced at, not confirmed either way).
- A generic manual-tracking fallback SDK for frameworks Cascaid doesn't
  auto-detect (AutoGen, hand-rolled orchestration).
- Additional vector DB adapters (Chroma/Qdrant/Milvus/LanceDB).
- The entire "Production readiness" section below (migrations, retention
  policy, restart policies/healthchecks, model-config versioning, CORS,
  secrets-at-rest, backup/restore docs) -- not touched this round at all,
  since the user asked to focus on compatibility first.

Start there next time rather than re-deriving this list from scratch.

## Session status (2026-09-02) — everything from the 2026-08-31 list is done; here's what's actually left

Checked each item on the "not started yet" list above against current code
rather than assuming it's still open:

- Direct OpenAI SDK instrumentation, coexisting with litellm dedup — **done**
  (`openai_adapter.py`).
- Direct Gemini SDK instrumentation — **done** (`gemini_adapter.py`).
- Generic manual-tracking fallback SDK for undetected frameworks — **done**
  (`manual_adapter.py`, PR #57, ADR 0002).
- Chroma/Qdrant/Milvus/LanceDB vector adapters — **done**
  (`vector_query_adapter.py`).
- Alembic migrations, retention job, restart policies/healthchecks, CORS,
  secrets-at-rest, backup/restore docs — **all done** (PRs #53-#55, ADRs
  0003-0005, `Backup_And_Restore.md`).

Reopened the underlying question this whole doc exists for — "is Cascaid
usable across the range of real pipelines/tool choices/edge cases a customer
might actually have" — now that the known bug list is empty. Re-derived the
gap list from current code (not from memory of this doc), ranked by real
impact:

1. **AutoGen has no native, auto-detected instrumentation.** The PRD names
   LangGraph/CrewAI/AutoGen together as the three auto-detected orchestrators
   (Section 5.2, 6.1) — `stack_detector.py`'s `ORCHESTRATOR_MODULES` only has
   `langgraph`/`crewai`. An AutoGen pipeline today gets silent zero topology
   extraction unless the customer already knows to reach for the manual
   fallback SDK — which works, but breaks the "zero code changes" promise
   Section 4.1 makes for the frameworks Cascaid claims to support out of the
   box. Largest remaining gap against the PRD's own stated integration list.
2. **Model-config/version metadata doesn't exist.** `train.py`'s
   `torch.save(gnn.state_dict(), out_path)` saves only weights, no
   hyperparameters (`hidden`/`layers`/`conv`) — confirmed still true, no
   `--layers`/`--conv` CLI flags exist anywhere. `serve`/`retrain` agree by
   accident today; the moment a customer's hyperparameters drift from a
   default, loading fails with a raw PyTorch state-dict shape error instead
   of a clear message. Contained, well-scoped fix.
3. **Contextvar-based attribution is process-local** (confirmed —
   `runtime_context.py` is plain `contextvars.ContextVar`, no cross-process
   propagation). A pipeline fanning work across Celery/Ray/multiprocessing
   workers loses attribution partway through. Real gap, but a genuine
   architecture decision (which backend(s) to support, how to propagate a
   context across an arbitrary worker boundary) — flagging for a scoping
   conversation, not starting speculatively, same reasoning the prior session
   used for the SDK adapters before building them.
4. **LangSmith/Phoenix trace-export import** — not re-opening this: the code
   comments (`langfuse_import.py`) already record this as a deliberate MVP
   scope decision (Langfuse picked as the one representative import path),
   not an oversight.

**#1 shipped this session.** `autogen_adapter.py` targets `autogen-agentchat`
(not the `ag2` fork — see `docs/adr/0006-autogen-agentchat-not-ag2.md` for why,
verified against the real installed package before writing any adapter code).
Two seams: `BaseGroupChat.__init__`/`run_stream` for static topology + step
tracking (every team type — RoundRobinGroupChat/SelectorGroupChat/Swarm/
MagenticOneGroupChat/GraphFlow — shares this base), and
`ChatAgentContainer.handle_request` for per-agent runtime tracking (the one
seam every team type dispatches a participant's actual turn through).
Contextvar propagation across `autogen_core`'s actor-runtime dispatch was
verified empirically with a standalone repro before trusting it — unlike
LangGraph's `ainvoke`, CrewAI's `async_execution` thread, or litellm's
background dispatch, it turned out to propagate correctly by default (plain
awaited coroutines within one asyncio task), so no CrewAI-style stash-on-object
workaround was needed. A real, if narrower, footgun did surface: naively
replacing `ChatAgentContainer.handle_request` silently drops it from
`autogen_core`'s `@event` handler-discovery (routing metadata lives in the
function's `__dict__`, re-read fresh per instance) — caught before shipping via
the same repro, fixed with `functools.wraps`. `stack_detector.py`'s
`ORCHESTRATOR_MODULES` became a label→module dict (matching
`VECTOR_DB_MODULES`/`DIRECT_SDK_MODULES`) since `autogen-agentchat` imports as
`autogen_agentchat`, not `autogen`. Full suite green twice consecutively (335
passed, 3 skipped both runs).

**#2 shipped too.** New `models/model_config.py`: a `ModelConfig` sidecar
(`in_dim`/`edge_dim`/`hidden`/`layers`/`conv`) saved as JSON next to the `.pt`
weights (`out_path.with_suffix(".config.json")`) -- the same sibling-file
convention `train.py` already used for `.drift_reference.json`, not a change
to the weights file's own format. `cascaid.train`/`cascaid.retrain` gained
`--hidden`/`--layers`/`--conv` CLI flags (train.py didn't have `--hidden` at
all before this; `--layers`/`--conv` existed nowhere) and now write the
sidecar alongside every model they save. `cascaid.serve`'s
`--hidden`/`--layers`/`--conv` default to unset rather than baked-in values:
`serving/risk.py`'s `load_model()` reads the sidecar automatically when
present, so a model trained with non-default hyperparameters (exactly what
`scripts/gnn_experiment.py`'s own accuracy sweeps produce) now loads with
zero flags, and an explicit flag that actually conflicts with the sidecar
raises a clear `ValueError` naming both values instead of failing deep inside
`load_state_dict` with a raw PyTorch tensor-shape mismatch. No sidecar (a
`.pt` predating this feature) falls back to today's exact defaults --
backward compatible, no forced migration. New e2e regression test proves the
real scenario end to end: train with `--hidden 16 --layers 3 --conv gat`,
serve with zero override flags, confirm it actually serves risk scores rather
than crashing. Full suite green twice consecutively (343 passed, 3 skipped
both runs, up from 335 -- 8 new tests).

**Not started this session, pick up next**: #3, distributed attribution
across Celery/Ray/multiprocessing -- still a scoping conversation (which
backend(s) to support, how to propagate context across an arbitrary worker
boundary), not a same-session build.

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
`CustomLogger` + metadata-snapshot mechanism.

**A third bug, found while checking whether the "sync path is reliable"
claim above actually held for every sync call shape**: it didn't for
*streaming* (`stream=True`). Verified empirically: litellm dispatches a
streaming sync `completion()` call's success callback via a background
`ThreadPoolExecutor` -- the exact same contextvar-loss failure mode as
`acompletion`'s deferred dispatch and CrewAI's `async_execution` thread,
just via yet another mechanism -- so ambient `current_run_id`/`current_step`
reads silently dropped every streaming CallEvent even on the path documented
above as "proven fast and reliable." litellm also fires one success callback
per streaming *chunk* plus a final aggregated one; naively fixing just the
attribution would have produced a run of near-duplicate, degenerate-latency
events per logical call instead of one. Fixed by switching the sync path to
read from the same metadata snapshot the async path already used (removing
the sync/async attribution split entirely -- only the *registration*
mechanism still differs, per the reasoning above), and skipping every
callback until the one where `kwargs["complete_streaming_response"]` is set.

All three bugs (and the sync/async dispatch-mechanism split) are covered by
deterministic or real-dispatch unit tests -- see
`tests/unit/test_langgraph_adapter.py` and
`tests/unit/test_litellm_adapter.py`. I'm fixing all of this directly rather
than just flagging it, since it's a correctness bug in already-shipped
functionality, not a scope or design decision.

## Pipeline compatibility: what else breaks or is silently unsupported

Ranked by how many real pipelines each one likely affects:

1. **Direct OpenAI/Anthropic/Gemini SDK calls are invisible to Cascaid.**
   Confirmed precisely, not just asserted: `stack_detector.py`'s
   `detect_stack()` hardcodes `model_gateway = "litellm" if
   is_available("litellm") else None` -- there is no branch that checks for
   `openai`/`anthropic`/`google-generativeai` at all. A pipeline calling
   `openai.chat.completions.create(...)` or `anthropic.messages.create(...)`
   directly -- arguably *more* common than routing every call through
   LiteLLM as a gateway -- gets `model_gateway=None` and silently no
   model-endpoint observability whatsoever, with no error, warning, or any
   signal to the customer that this is happening. This is probably the
   single largest "will this work on my pipeline" gap, larger than the
   orchestrator-framework question below, because it affects LangGraph and
   CrewAI pipelines alike regardless of which orchestrator is used. Building
   real coverage here (new adapters, one per SDK, plus a design decision on
   how much to unify with the litellm adapter's shape) is a genuine new-build
   task, not a same-session bug fix -- flagged for a scoping conversation
   rather than started speculatively.
2. **~~CrewAI's `kickoff_async` isn't patched~~ -- checked precisely, and the
   real gap was worse and different.** `Crew.kickoff_async` itself turned
   out fine unpatched: it's `return await asyncio.to_thread(self.kickoff, ...)`,
   and `asyncio.to_thread` *does* copy contextvars into the new thread, so it
   correctly reaches the already-patched `Crew.kickoff`. The real bug is one
   level down: any `Task(async_execution=True)` -- a real, commonly-used
   CrewAI feature for parallelizing independent tasks *within* a normal
   `Crew.kickoff()` run (`crew.py`'s `if task.async_execution:` branch) --
   runs via `Task.execute_async`, which spawns a raw `threading.Thread` (not
   `asyncio.to_thread`). A raw thread does not inherit contextvars, so the
   async task's `run_id`/`step` read as `None` inside it (dropping every
   LiteLLM/vector-DB CallEvent it makes), and its node-name attribution
   silently fell back to the wrong task's name (verified against real
   CrewAI: a 3-task crew with the middle task async attributed it as
   `"researcher (0)"` instead of `"researcher (1)"` -- wrong, not just
   missing). Fixed the same session this was found: stash
   `run_id`/`step`/`name` as attributes directly on the live `Task` object
   at kickoff time (plain attribute access needs no thread-context
   propagation) instead of relying on contextvars alone.
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

## What got fixed this round vs. flagged for a decision

**Fixed** (bug fixes in existing code, no design decisions, no public API
changes) -- see the session-status note at the top for the full list and
PR numbers: LangGraph `ainvoke`, CrewAI `async_execution`, Pinecone method
drift, Weaviate's async client, LiteLLM streaming.

**Flagging, not building yet** -- each of these is either a genuine
infra/architecture decision (worth a quick confirmation before sinking time
into it) or a larger net-new build than a same-session bug fix:
- Direct OpenAI/Anthropic/Gemini SDK instrumentation (a new adapter, same
  shape as the existing LiteLLM one, but real new-build effort) -- the
  single largest remaining compatibility gap
- A generic manual-tracking fallback SDK for unsupported frameworks
- Additional vector DB adapters (Chroma/Qdrant/Milvus/LanceDB)
- Introducing Alembic migrations (a real workflow change for every future
  schema edit, not just this one)
- Designing and shipping an actual retention policy (TimescaleDB
  hypertables vs. a simpler periodic-delete job vs. just documenting a
  customer-side cron -- genuine trade-off, not obvious)
- `docker-compose.yml` restart policies + serve/dashboard healthchecks
  (small, but changes how the stack behaves operationally, worth a nod
  before changing)
