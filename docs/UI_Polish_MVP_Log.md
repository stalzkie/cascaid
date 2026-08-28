# UI Polish for MVP Testing — Log

Follow-up to `MVP_Accuracy_and_Product_Roadmap.md`'s UI plan section, which
identified the two biggest "feels unfinished" gaps in the dashboard (PR #17):
no way to discover a run without already knowing its id, and no live
refresh. This closes both, plus the smaller polish that fell out of doing
them properly (auto-refresh toggle, last-updated indicator).

Deliberately **not** in scope here (see the roadmap doc): auth, Grafana
panel, MCP server exposure. Those are the next sequencing step, not part of
"good to test as an MVP."

## What changed

### Backend: `GET /runs`

The dashboard could only ever show a run if you already knew its id and
typed it in — there was no way to discover what runs exist. Added the
missing layer, TDD'd end to end:

- `cascaid.ingestion.graph_store.list_runs(store_dir)` — lists run_ids with
  at least one persisted snapshot (a run subdirectory under the store).
  This is the same definition of "a run exists" that `latest_snapshot()`
  already uses, so it can't disagree with what `/pipeline/{run_id}` would
  actually return.
- `cascaid.dashboard.views.list_runs_view()` — thin wrapper, kept for
  consistency with the module's existing view-per-endpoint shape.
- `cascaid.dashboard.api`: `GET /runs` → `{"run_ids": [...]}`, sorted.

Covered by a unit test on `list_runs`, a unit test on the view, an
integration test on the endpoint (real FastAPI app + real graph store), and
an assertion added to the existing real-pipeline e2e test
(`test_dashboard_cli.py`) rather than a whole new e2e test — the pipeline
already existed, it just wasn't asserting on `/runs`.

### Frontend: run picker, auto-refresh, last-updated

- **Run picker**: a "Known runs" section above the manual text-entry form,
  fetched from `GET /runs` on mount and re-polled every 10s so it stays
  current as new runs appear (e.g., a live pipeline connecting for the
  first time). Clicking a run loads it — the manual text input stays as a
  fallback for typing an id that isn't in the list yet. Empty state shows a
  specific hint ("if you just ran `docker compose up`, the seed step may
  still be finishing") rather than a bare empty list, since that's the
  most likely reason it's empty for a first-time tester.
- **Auto-refresh**: while a run is loaded, the pipeline + track record
  re-fetch every 10s (toggle-able, on by default) instead of requiring a
  manual "Refresh" click to see risk scores update.
- **Last updated**: a small timestamp shown next to the loaded content, so
  it's visible at a glance whether auto-refresh is actually working rather
  than silently stuck.

New/updated frontend tests: `fetchRuns` in `api.test.ts` (success + error
paths), and three new `App.test.tsx` cases (empty-run-list state, run list
rendering + click-to-load, and asserting the last-updated text appears
after a successful load) alongside updating the two pre-existing tests'
fetch mock to also answer `/runs`.

## Environment note (same constraint as every prior frontend change)

No node/npm in this sandbox, so this could only be validated through CI
(the `frontend` job: typecheck, vitest, build) — same as the original
frontend PR (#17) and the deployment work (#18). No surprises hit this
time; typecheck and tests passed on the first CI run.

## Verification

- [x] `pytest tests/unit/test_graph_store.py`, `test_dashboard_views.py` —
  `list_runs`/`list_runs_view` round trip
- [x] `pytest tests/integration/test_dashboard_api.py` — real `GET /runs`
  via TestClient
- [x] `pytest tests/e2e/test_dashboard_cli.py` — `/runs` assertion added to
  the existing real-pipeline e2e test
- [x] Full Python `tests/unit`/`tests/integration`/`tests/e2e` pass
- [x] `npm run typecheck`, `npm test`, `npm run build` — via CI's
  `frontend` job (see note above)
- [x] `docker` CI job — full stack still builds and boots with these
  changes (dashboard container serves the new endpoint, frontend container
  builds against it)
