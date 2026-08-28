# Auto-Instrumentation Glue Layer Plan (2026-08-28)

Plan for closing the gap between the PRD's "zero instrumentation, one command"
pitch (§4.1, §4.4) and what actually exists today, so beta testers get an
install experience closer to graphify's (`graphify query "..."` — one
command, auto-detects the codebase, immediate value) or the one-command
setup other dev tools use, instead of hand-wiring adapters themselves.

## Two install surfaces — don't conflate them

Cascaid has two genuinely different things a user installs, and this plan
only concerns the second one:

1. **The Cascaid stack** (Postgres, model server, dashboard API, frontend).
   Already solved: `docker compose up` (PRD 4.4). Not touched by this plan.
2. **The instrumentation agent** — the thing that has to go *inside the
   customer's own pipeline process* to observe it. This is the actual gap.
   Nothing wires it up automatically today.

## Current state (verified against the code, not the PRD's claims)

- `stack_detector.py` — real, works: checks `importlib.util.find_spec` for
  `langgraph`, `litellm`, `pinecone`/`weaviate`/`pgvector`. This is the
  entire "auto-detect" mechanism.
- `langgraph_adapter.py` — real: `extract_static_topology(compiled_graph)`
  reads node/edge structure from an already-compiled LangGraph app. Requires
  the customer to hand Cascaid that compiled object; nothing calls this
  automatically.
- `litellm_adapter.py` — real: converts a LiteLLM callback payload into a
  `CallEvent`. Nothing registers these functions as LiteLLM
  `success_callback`/`failure_callback` anywhere in `src/`.
- `vector_query_adapter.py` — real: `observe_vector_query(...)` is a context
  manager the customer's code must explicitly wrap around a query call.
  Nothing patches Pinecone/Weaviate/pgvector client methods automatically.
- `runtime_context.py` — real: `track_node(name)` sets a contextvar so a
  model call can be attributed to the LangGraph node that made it. Nothing
  calls this automatically around node execution.
- **None of the above are wired together.** Even Cascaid's own demo pipeline
  (`cascaid_demo/pipeline.py`) doesn't use these adapters — it calls a
  hand-written fake `recorder.log(...)` instead. The adapters are only
  exercised by their own unit tests today.
- **There is no `cascaid` command.** No `[project.scripts]` entry in
  `pyproject.toml`. Every entrypoint (`cascaid.serve`, `cascaid.train`,
  `cascaid.dashboard.serve`, `cascaid_demo.run_scenarios`) is invoked as
  `python -m cascaid.xxx` with its own separate `argparse` setup. `cascaid
  demo` from PRD 4.2 doesn't exist as a literal command.
- `litellm` is a **dev-only** dependency in `pyproject.toml`
  (`[dependency-groups] dev`), not a runtime dependency of the installable
  package — correctly so, since Cascaid should detect it in the *customer's*
  environment, not bundle it. Worth double-checking this stays lazy/optional
  once real imports of `litellm`-specific types show up in the patching code.

## Design goal

One command that turns "nothing" into "Cascaid is watching this pipeline,"
with no manual wiring of adapters, no `track_node()` calls added to their
code, no callback registration by hand.

## Proposed seam: `cascaid run -- <their command>`

Modeled on `ddtrace-run` / `opentelemetry-instrument` — proven prior art for
exactly this problem (auto-instrument a running Python process without
editing its source). Not a novel idea; a known-working pattern.

```
cascaid run -- python app.py
cascaid run -- uvicorn app:api --reload
```

Interface: one subcommand, one argument (the customer's normal launch
command, unchanged). Everything else is implementation, invisible to the
caller — that's the deep-module shape this needs: small interface, all the
patching complexity hidden behind it.

Behind that one line, at process start (before the customer's app module
loads):

1. Run `detect_stack()` (already built) to decide which patches to apply.
2. **LangGraph**: monkey-patch `StateGraph.compile` so every compiled graph
   is automatically run through `extract_static_topology()` and registered,
   and monkey-patch the compiled graph's node-execution path so each node
   call is automatically wrapped in `track_node(name)` — the customer never
   calls this themselves.
3. **LiteLLM**: **append** to `litellm.success_callback` /
   `litellm.failure_callback` rather than overwrite them — a customer who
   already has Langfuse/LangSmith registered there must keep getting their
   existing traces; Cascaid composes, doesn't clobber.
4. **Vector DBs**: patch the known query methods on the client libraries
   `stack_detector` found (`pinecone.Index.query`,
   `weaviate` client's query method) to auto-wrap them in
   `observe_vector_query`. **pgvector is scoped out of full auto-patch for
   the beta** — it's not a distinct client library (it's a Postgres
   extension invoked through psycopg/SQLAlchemy), so reliably detecting
   "this query is a vector similarity search" without false positives needs
   more design than the other two. Document it as "add one `with
   observe_vector_query(...):` line around your pgvector query" for beta
   users on that stack, rather than pretending it's automatic when it isn't.
5. Stream resulting `CallEvent`s into the existing Graph Store /
   `snapshot_builder.py` path, pointed at the local Cascaid stack instead of
   demo data.

## Packaging work needed to make "one command" literal

- Add `[project.scripts] cascaid = "cascaid.cli:main"` to `pyproject.toml`,
  with subcommands `run`, `demo` (wraps `cascaid_demo.run_scenarios` +
  friends into the single command PRD 4.2 already promises), `serve`,
  `dashboard`. This replaces the current pile of separate `python -m`
  invocations with one binary, matching how graphify presents as a single
  `graphify <verb>` command rather than several scripts.
- **Decided: publish to public PyPI.** `pipx install cascaid` /
  `uv tool install cascaid` gives testers a real `cascaid` binary on PATH —
  no private channel, no `git+https://...` install link. The goal is
  testers never run `uv sync` + `python -m` chains by hand.

## Beta tester's actual golden path once this lands

```
pipx install cascaid          # or: uv tool install cascaid
cascaid demo                  # zero-risk first contact, synthetic pipeline (PRD 4.2)
docker compose up             # stand up the local Cascaid stack (already works today)
cascaid run -- python app.py  # point it at their real pipeline, no code changes
```

Then the existing UI flow (observe-only → dashboard → opt-in alerting)
already described in the roadmap discussion applies unchanged.

## Explicitly out of scope for this beta pass

- Auto-instrumenting CrewAI/AutoGen — only a LangGraph adapter exists today;
  extending the same monkey-patch approach to other orchestrators is a
  separate, later effort once the LangGraph path is proven.
- Full automatic pgvector detection (see above) — ship the documented
  one-line manual wrap for beta, revisit auto-patching post-beta.
- Windows/process-injection edge cases beyond the standard CPython import
  hook approach `ddtrace-run` itself relies on.

## Suggested build order

1. `cascaid.cli` module + `[project.scripts]` entry, wrapping *existing*
   commands (`demo`, `serve`, `dashboard`) under one binary — ships
   immediately, zero new instrumentation risk, and alone already improves
   the install story.
2. Wire the LiteLLM adapter for real: auto-register (appending, not
   replacing) callbacks when `cascaid run` starts and LiteLLM is detected.
   Lowest-risk patch target since it uses LiteLLM's own callback API rather
   than monkey-patching internals.
3. Wire the LangGraph adapter for real: patch `StateGraph.compile` +
   node-execution wrapping. Prove this against Cascaid's own demo pipeline
   first (replace the fake `recorder.log` there with the real adapters —
   this alone would catch integration bugs the adapters' isolated unit
   tests can't).
4. Vector DB auto-patch for Pinecone/Weaviate (pgvector stays manual per
   above).
5. Beta packaging: publish, write the four-line golden-path README section
   above to replace the current dev-only `uv sync` instructions.

## Decisions (2026-08-28)

- **Distribution: public PyPI.** No private/TestPyPI staging channel — the
  first public install is also the first beta tester's install. This raises
  the bar on step 3 below: the instrumentation code has to be proven before
  anyone outside can run `pip install cascaid`, since there's no private
  gate left to catch a broken patch first.
- **Migrate the demo pipeline onto the real adapters before recruiting any
  beta tester.** `cascaid_demo/pipeline.py`'s fake recorder gets replaced
  with `litellm_adapter`/`langgraph_adapter`/`runtime_context`/
  `vector_query_adapter` (build-order step 3) and proven working end-to-end
  first. Combined with the public-PyPI decision above, this migration is
  now the load-bearing gate before publishing at all, not an optional
  hardening pass — publish is blocked on it.
