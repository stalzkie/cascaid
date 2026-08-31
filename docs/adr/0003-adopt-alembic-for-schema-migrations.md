---
status: accepted
---

# Adopt Alembic for schema migrations now, rather than deferring

Cascaid's schema (`src/cascaid/storage/models.py`) is created today via
`Base.metadata.create_all` -- fine for a fresh install, but any existing customer
deployment can't pick up a schema change (a new column, a new table) without either
losing data or someone hand-patching their database. The schema already has five tables
(`Config`, `AuthSession`, `AlertHistory`, `IncidentLabel`, `ScoreHistory`) and is still
growing (this same round of work adds retention and secrets-at-rest changes, see ADR
0004 and ADR 0005).

We're adopting Alembic now rather than waiting for a real customer upgrade to force the
issue: every table that exists before Alembic is adopted needs a baseline migration
capturing it as a starting point, so the earlier this happens, the fewer schema versions
there are to reconstruct that baseline from. Deferring doesn't avoid the migration
tooling, it just adds more schema history for the eventual baseline to cover.

## Considered Options

- **Defer** (rejected): keep `create_all`-only until a real customer needs an in-place
  upgrade. Reversible in the sense that Alembic can be added at any point, but every
  schema change made in the meantime becomes something the eventual baseline migration
  has to account for after the fact instead of being captured as its own revision when
  it happened.

## Consequences

- Every future schema change (columns, tables, indexes) goes through an Alembic
  revision from now on -- a workflow change for whoever edits `models.py` next, not
  just a one-time setup step.
- A baseline revision needs to exist that matches the current `create_all` output
  exactly, so existing fresh installs and new Alembic-managed installs converge on the
  same schema.
