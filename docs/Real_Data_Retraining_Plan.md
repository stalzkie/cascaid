# Scheduled Retraining on Real Customer Data — Design Note (2026-08-30)

Follow-up to `Client_Readiness_and_YC_Grade_Assessment.md` §3.3. Investigated
before writing any training code, because what I found changes the scope of
that item from "wire an existing trigger to an existing pipeline" to
"a schema decision that touches the on-disk/JSONL format a published PyPI
package already writes." Flagging it rather than guessing, per the
domain-modeling skill's ADR criteria: hard to reverse, surprising without
context, and a genuine trade-off.

## What I found

`cascaid.train`'s pipeline labels a node-step positive using
`fault_onset_step`/`cascade_step` from the synthetic scenario manifest
(`labeling.py`'s `label_step()`). Neither concept exists for real customer
data — an `IncidentLabel` row (`storage/models.py`) is just
`(run_id, node_name, incident_type, occurred_at, source)`, with no "onset"
or "cascade" step to build a ramp window from.

That gap alone would just mean "write a different labeling function for
real data" (the point-in-time-window approach below) -- a normal, bounded
piece of work. The actual blocker is one level down: **`CallEvent`
(`ingestion/schema.py`) and `Snapshot` (`ingestion/snapshot_builder.py`)
carry no wall-clock timestamp at all** -- a snapshot's only time coordinate
is an integer `step` (sequential invocation count within one run). There is
currently no way to answer "which snapshot step was active when this
`IncidentLabel.occurred_at` happened" for a real run, because nothing
records when a step *occurred*, only its order.

This is invisible in the synthetic path because the manifest already states
`fault_onset_step`/`cascade_step` directly in step-index terms -- it never
needed wall-clock time. It becomes load-bearing the moment labels come from
an external system (Langfuse, a manual incident report) that only knows
wall-clock time.

## Why this is a real decision, not just a bug

`CallEvent` is the wire format between `cascaid run`'s bootstrap sink and
`cascaid ingest --follow` (JSON-lines on disk) -- the same format
`cascaid==0.2.0` already ships to the public. Adding a required field
would break any JSONL log a beta tester has already captured; adding an
optional field is backward-compatible but means real event logs newer than
this change have a field that all synthetic/demo data won't, which the
training pipeline needs to treat as optional forever, not clean up later.

## Proposed shape (for review, not yet built)

1. Add `occurred_at: datetime` to `CallEvent`, optional (defaulting to
   `None` and handled by `from_json` for old logs without it) -- populated
   by the adapters (`litellm_adapter.py`, `vector_query_adapter.py`) from
   `datetime.now(timezone.utc)` at observation time, not backfilled.
2. Carry the first/last `occurred_at` seen per step into `Snapshot` (a
   `[step_start, step_end)` wall-clock interval), so a snapshot step can
   answer "was this IncidentLabel's `occurred_at` inside my window."
3. New labeling function alongside (not replacing) `label_step()`:
   `label_step_from_incidents(node_order, incident_labels, step_start,
   step_end, window_before, window_after)` -- marks a node positive if an
   `IncidentLabel` for that node falls within
   `[step_start - window_before, step_end + window_after)`. **Deliberately
   node-local only** -- no predecessor-propagation the way
   `affected_nodes()` does for synthetic scenarios, because that
   propagation rule was calibrated against synthetic ramp semantics
   (`GNN_Accuracy_Improvement_Log.md`), and assuming a real incident
   cascades to callers the same way would be an unverified guess baked
   into every customer's model.
4. `cascaid retrain --database-url ... --store ... --out ...`: builds a
   dataset from Graph Store snapshots + `IncidentLabel` rows for a live
   install (via step 3's labeling function), retrains, evaluates against a
   held-out split of the *real* runs (not the synthetic held-out set --
   a separate number, reported separately), and only overwrites the served
   model artifact if the new model's PR-AUC on that real held-out split is
   at or above a configurable floor (protects against a small, noisy batch
   of real incidents silently making the served model worse than the
   pretrained base). Manual/cron-triggered first; wiring it to fire
   automatically off `drift.py`'s existing PSI check is a natural
   fast-follow once the manual command is proven -- not a prerequisite.

## What I need from you

Whether to make the `CallEvent`/`Snapshot` schema change described above.
It's backward-compatible (additive, optional field) but it's still a
change to a format already in a public release, and it's the actual
prerequisite for real-data retraining -- everything else in this plan is
normal engineering once that's decided. I didn't implement it without
checking first.
