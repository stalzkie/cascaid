---
status: accepted
---

# Retention for incident/score history is a periodic-delete job built into Cascaid, not TimescaleDB or a documented customer-side cron

Incident and score history (`IncidentLabel`, `ScoreHistory`, `AlertHistory`) grows
without bound today -- nothing expires it. Three real options exist, evaluated against
Cascaid's actual userbase (self-hosted customers running in their own VPC, per the
existing CORS rationale in `dashboard/api.py`) rather than against what's easiest to
build:

TimescaleDB hypertables are the most "correct" long-term answer at real scale, but they
require either a customer's Postgres to support the Timescale extension or Cascaid to
ship its own Postgres image -- a real deployment blocker for the likely-common case of a
customer on a managed Postgres (RDS, Cloud SQL) that doesn't support arbitrary
extensions, and disproportionate infra lock-in for a product that's currently alpha and
not yet operating at a scale where it matters.

A documented customer-side cron/SQL snippet costs nothing to build, but silently depends
on every self-hosting customer actually setting it up -- realistically, most won't until
history has already grown large enough to hurt, turning retention into a support
incident instead of something that just works. That's a worse fit for a product whose
whole premise (`_instrument_bootstrap.py`) is minimizing customer effort.

We're building a periodic-delete job inside Cascaid instead: a background task
(no new container) in the existing `serve` process, deleting rows older than a
configurable retention window (`Config`/env var, no new table). It works against any
Postgres a customer already has, doesn't ask them to build or maintain anything, and
doesn't front-load an infra dependency ahead of actual need.

## Considered Options

- **TimescaleDB hypertables** (rejected): most scalable long-term, but couples
  retention to a specific Postgres extension/image the self-hosted userbase can't
  always provide.
- **Documented customer-side cron** (rejected): zero build cost, but shifts a
  correctness-affecting responsibility onto the customer with no guarantee they act on
  it.

## Consequences

- Needs a scheduler primitive inside the `serve` process that doesn't exist yet (this
  codebase has no background-job runner today) -- new, if small, infrastructure.
- The retention window is configurable but Cascaid now owns a default; picking too
  short a default risks deleting history a customer still wanted (e.g. for drift
  comparison), so the default needs to be conservative and documented, not just chosen
  for storage efficiency.
