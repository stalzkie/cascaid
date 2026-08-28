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
2. **LangGraph** (built 2026-08-28, corrected from the original design below):
   `StateGraph.add_node` is patched — not LangGraph's internal Pregel
   dispatch loop — to wrap each node's function in `track_node(name)` at
   graph-construction time. A smaller, more targeted seam than intercepting
   internal execution, and far less coupled to a specific LangGraph version.
   `StateGraph.compile` is patched separately to extract topology once via
   `extract_static_topology()` and wrap the compiled graph's
   `invoke`/`ainvoke` in a fresh `track_step` per call, so `step` keeps
   meaning "one top-level invocation" — matching what the GNN was actually
   trained on (see `runtime_context.py`'s `current_run_id`/`current_step`,
   sibling to `current_node`). Fragmenting one invocation across several
   call-indexed steps would have been a train/serve distribution mismatch,
   not just an integration shortcut.
3. **LiteLLM**: **append** to `litellm.success_callback` /
   `litellm.failure_callback` rather than overwrite them — a customer who
   already has Langfuse/LangSmith registered there must keep getting their
   existing traces; Cascaid composes, doesn't clobber.
4. **Vector DBs** (built 2026-08-28, corrected from the original design
   below): neither vendor has "the query method" (singular) — verified via
   introspection against the installed packages, not assumed. **Pinecone**
   `Index` has 4: `query`, `query_namespaces`, `search`, `fetch`. **Weaviate**
   `Collection.query` (really `_QueryCollection`) has 10:
   `near_vector`/`near_text`/`near_object`/`near_image`/`near_media`/
   `hybrid`/`bm25` (retrieval-shaped) plus `fetch_objects`/
   `fetch_object_by_id`/`fetch_objects_by_ids` (plain lookups). Decided to
   patch **all of them, every method, on both vendors** — patching only the
   obvious one (`near_vector`) would silently under-count real retrieval
   activity, understating vector-store load to the GNN: an accuracy problem,
   not just a coverage gap. `register_pinecone_callbacks`/
   `register_weaviate_callbacks` in `vector_query_adapter.py` reuse the
   existing `observe_vector_query` converter the same way
   `register_litellm_callbacks` reuses its converters. **Testing limitation,
   not a shortcut**: neither SDK has an offline/mock dispatch mode like
   LiteLLM's `mock_response` (no local emulator), and `cascaid run` patches
   `Index.query` etc. *before* a target script runs — so a target script
   can't install a stand-in afterward without clobbering the wrapper, and
   there's no live backend to test against for real. Proven instead at the
   unit level against the real classes/method names (introspected, not
   guessed) with a stand-in substituted only at the innermost network-call
   layer — the one thing that genuinely can't be exercised offline.
   **pgvector is scoped out of full auto-patch for the beta** — it's not a
   distinct client library (it's a Postgres extension invoked through
   psycopg/SQLAlchemy), so reliably detecting "this query is a vector
   similarity search" without false positives needs more design than the
   other two. Document it as "add one `with observe_vector_query(...):` line
   around your pgvector query" for beta users on that stack, rather than
   pretending it's automatic when it isn't.
5. Stream resulting `CallEvent`s into the existing Graph Store /
   `snapshot_builder.py` path, pointed at the local Cascaid stack instead of
   demo data. **Still open** (2026-08-28): `cascaid run` (built) currently
   writes topology + `CallEvent`s as JSON lines to a local file
   (`CASCAID_EVENTS_PATH`, default `data/live/<run_id>.jsonl`) via
   `cascaid/_instrument_bootstrap.py` — proven end-to-end against a real
   subprocess (`tests/e2e/test_run_instrumented.py`), but nothing yet reads
   that file into the Graph Store/Postgres, so a beta tester can observe
   their pipeline but can't see it in the dashboard yet. That wiring is not
   done.

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

1. ✅ **Done.** `cascaid.cli` module + `[project.scripts]` entry, wrapping
   *existing* commands (`demo`, `serve`, `dashboard`) under one binary.
   PR #25.
2. ✅ **Done.** Wire the LiteLLM adapter for real:
   `register_litellm_callbacks(sink)` appends (never replaces) to
   `litellm.success_callback`/`failure_callback`; added
   `current_run_id`/`current_step` to `runtime_context.py`. PR #26.
3. ✅ **Done** (2026-08-28), all three parts in one pass:
   - `instrument_langgraph(topology_sink)` — the `add_node`/`compile` patch
     described above.
   - `tests/integration/test_instrumentation_integration.py` — the
     publish-blocking proof, against a dedicated real pipeline (not
     `run_scenarios.py`, see the superseded decision above).
   - `cascaid run -- <command>` — a real subprocess launcher
     (`cascaid/_instrument_bootstrap.py` + a generated `sitecustomize.py`
     prepended to `PYTHONPATH`, the same trick `ddtrace-run` uses), proven
     against an actually-launched subprocess in
     `tests/e2e/test_run_instrumented.py`. Events currently land in a local
     JSON-lines file, not the Graph Store — see "Still open" above.
4. ✅ **Done** (2026-08-28). Vector DB auto-patch for Pinecone (4 methods)
   and Weaviate (10 methods) — see above. `pinecone`/`weaviate-client` added
   as dev-only dependencies (mirrors `litellm`'s pattern). Wired into
   `cascaid/_instrument_bootstrap.py` so `cascaid run` applies them
   automatically when detected.
5. Beta packaging: publish, write the four-line golden-path README section
   above to replace the current dev-only `uv sync` instructions. Not
   started — and per the "Still open" note above, `cascaid run`'s events
   need to reach the Graph Store before this is genuinely beta-ready, not
   just publishable.

## Decisions (2026-08-28)

- **Distribution: public PyPI.** No private/TestPyPI staging channel — the
  first public install is also the first beta tester's install. This raises
  the bar on step 3 below: the instrumentation code has to be proven before
  anyone outside can run `pip install cascaid`, since there's no private
  gate left to catch a broken patch first.
- **Migrate the demo pipeline onto the real adapters before recruiting any
  beta tester.** ~~`cascaid_demo/pipeline.py`'s fake recorder gets replaced
  with `litellm_adapter`/`langgraph_adapter`/`runtime_context`/
  `vector_query_adapter`~~ **Superseded 2026-08-28, see step 3 below**: the
  demo's fault injection is precise `rng`-controlled statistics validated
  against the 0.90 PR-AUC number — routing it through real LiteLLM dispatch
  would risk that number (no controllable latency/error/cost) for no
  integration-correctness benefit. `cascaid_demo/run_scenarios.py` and its
  mocks are untouched, permanently, not just for this pass. The
  publish-blocking gate is satisfied instead by a dedicated integration test
  (`tests/integration/test_instrumentation_integration.py`) proving the real
  adapters work together against a real (if small) LangGraph+LiteLLM
  pipeline built just for that purpose.
