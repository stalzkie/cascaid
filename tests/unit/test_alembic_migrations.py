"""Verifies the Alembic baseline (ADR 0003) produces the exact same schema as
repository.init_db's Base.metadata.create_all -- not by eyeballing the generated
revision, but by actually running both paths against a real (SQLite) database and
diffing the reflected schema. Both paths ultimately read from the same
cascaid.storage.models.Base.metadata (see alembic/env.py's target_metadata), so this
mostly guards against the baseline revision drifting out of sync with models.py after
a hand-edit -- Postgres-specific dialect behavior (timezone-aware timestamps,
autoincrement) is covered separately by tests/integration/test_storage_postgres.py,
gated on a real Postgres via CASCAID_TEST_DATABASE_URL in CI."""

from __future__ import annotations

import os
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from cascaid.storage.repository import init_db

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reflect_schema(database_url: str) -> dict[str, dict[str, str]]:
    engine = create_engine(database_url)
    inspector = inspect(engine)
    schema = {}
    for table_name in sorted(inspector.get_table_names()):
        if table_name == "alembic_version":
            continue
        columns = {col["name"]: str(col["type"]) for col in inspector.get_columns(table_name)}
        schema[table_name] = columns
    return schema


def test_alembic_baseline_matches_create_all_schema(tmp_path):
    create_all_db = tmp_path / "create_all.db"
    alembic_db = tmp_path / "alembic.db"

    init_db(create_engine(f"sqlite:///{create_all_db}"))

    original_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{alembic_db}"
    try:
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
        command.upgrade(cfg, "head")
    finally:
        if original_database_url is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = original_database_url

    assert _reflect_schema(f"sqlite:///{alembic_db}") == _reflect_schema(f"sqlite:///{create_all_db}")
