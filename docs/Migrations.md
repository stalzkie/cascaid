# Schema Migrations

Cascaid's schema (`src/cascaid/storage/models.py`) has two ways to reach its current
shape, and they're both meant to converge on the exact same result -- verified by
`tests/unit/test_alembic_migrations.py` (SQLite) and
`tests/integration/test_storage_postgres.py` (real Postgres, CI only):

- **`init_db()` / `Base.metadata.create_all`** (`cascaid.storage.repository.init_db`,
  used by every `cascaid` CLI entrypoint and the whole test suite): builds the full
  current schema from scratch. Fine for a **fresh install** -- a new database with no
  tables yet -- and is what SQLite-backed tests and the demo pipeline still use, since
  Alembic's autogenerate machinery is unnecessary overhead for a throwaway/fresh
  SQLite file.
- **`alembic upgrade head`**: applies schema changes to an **existing** deployment
  in place. This is what a real customer with data already in Postgres needs to run
  after upgrading Cascaid across a version that changed the schema -- `create_all` is a
  no-op against tables that already exist, so it will not pick up a new column or
  table on its own.

## For a customer upgrading an existing Postgres deployment

```
DATABASE_URL=postgresql+psycopg://... python -m alembic upgrade head
```

`alembic/env.py` reads `DATABASE_URL` the same way every other `cascaid` command does
(see `cascaid.storage.db.get_engine`) -- no need to edit `alembic.ini`.

## For anyone changing `models.py`

Every schema change (new column, new table, new index) from now on needs its own
Alembic revision:

```
DATABASE_URL=postgresql+psycopg://... python -m alembic revision --autogenerate -m "..."
```

Review the generated revision before committing it -- autogenerate is a starting point,
not a guarantee (it won't detect every kind of change, e.g. some constraint renames).
`init_db()`/`create_all` doesn't need touching when you do this; it always reflects
`models.py` directly and stays correct for fresh installs automatically.
