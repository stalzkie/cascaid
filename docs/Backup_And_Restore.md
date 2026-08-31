# Backup & Restore

Closes the gap flagged in `Production_Readiness_and_Pipeline_Compatibility_Assessment.md`
("No backup/restore documentation for the self-hosted Postgres data").

## What actually needs backing up

Only the Postgres database. Everything else in a `docker compose up` deployment is
either regeneratable or not real customer data:

- **`postgres-data` volume (back this up)** -- the `score_history`, `incident_labels`,
  `alert_history`, `config`, and `auth_session` tables (see
  `src/cascaid/storage/models.py`). This is the only place a customer's real,
  accumulated incident/score history lives. It cannot be regenerated -- losing it loses
  real operational history, not synthetic data.
- **`graph-store` volume (do not bother)** -- the latest graph snapshot per run, written
  by whatever pipeline is instrumented. If lost, it repopulates from the next real
  ingestion run; nothing history-bearing lives here permanently.
- **`models` volume (do not bother)** -- the trained/pretrained GNN checkpoint. The
  `seed` service regenerates a fresh one from synthetic scenarios on `docker compose up`
  if this volume is lost (see `docker-compose.yml`'s `seed` service).

## Backing up

Using `pg_dump` against the `postgres` service (matches this repo's
`docker-compose.yml` service name and `.env`/`.env.example` variable names):

```sh
docker compose exec postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F c -f /tmp/cascaid-backup.dump
docker compose cp postgres:/tmp/cascaid-backup.dump ./cascaid-backup-$(date +%Y%m%d).dump
```

`-F c` (custom format) is compressed and works with `pg_restore`'s `--clean`/`--if-exists`
options below, which a plain SQL dump doesn't support as cleanly.

## Restoring

Against a running `postgres` service with an empty (or to-be-overwritten) `$POSTGRES_DB`:

```sh
docker compose cp ./cascaid-backup-YYYYMMDD.dump postgres:/tmp/cascaid-backup.dump
docker compose exec postgres pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists /tmp/cascaid-backup.dump
```

Restart `serve`/`dashboard` afterwards (`docker compose restart serve dashboard`) so any
in-process state picks up the restored data.

## Backup schedule vs. retention

A separate retention policy (see the relevant ADR under `docs/adr/`, if merged) deletes
history older than a configured window. A backup schedule needs to run more frequently
than that retention window, or history can be deleted before it's ever captured in a
backup. If the retention window is, say, 90 days, a daily or weekly backup cron
comfortably stays ahead of it; a backup cadence longer than the retention window
defeats the point of taking backups at all. Pick your actual cadence based on the
retention window in effect for your deployment, not the example above.
