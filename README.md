# Cascaid

Predictive cascading-failure intelligence for AI-native systems: predicts which
part of your LangGraph/LiteLLM/vector-DB pipeline is about to take the rest
down with it, before it happens -- not just tracing what already went wrong.
Self-hosted, your data never leaves your own environment. See
`docs/Cascaid_PRD (1).md` for the full product spec.

## Install

```
pipx install cascaid          # or: uv tool install cascaid
```

## Try it -- zero setup, no real pipeline needed

```
cascaid demo
```

Spins up a synthetic fault-injection pipeline, trains a model on it, and seeds
a local SQLite-backed store -- no Postgres, no Docker, nothing to configure
first. See it predict a cascade before it happens.

## Point it at your real pipeline

```
docker compose up                                  # the dashboard + model server + Postgres, one command
cascaid run -- python your_app.py                   # instruments your app with zero code changes
cascaid ingest --events data/live/<run_id>.jsonl \
  --model models/pretrained_base.pt \
  --database-url postgresql+psycopg://... \
  --follow                                          # gets it showing up in the dashboard
```

`cascaid run` auto-detects LangGraph/LiteLLM/Pinecone/Weaviate in your app and
instruments them before your code runs -- no SDK, no decorators, no lines added
to your pipeline. `cascaid ingest` streams what it observes into the same
Graph Store + Postgres the dashboard reads from. Alerting is off by default;
turn it on once you trust the track record.

Then open `http://localhost:3000` for the dashboard.

## Development

```
uv sync                              # install pinned Python 3.12 env + deps
uv run python -m cascaid_demo.run_scenarios   # generate synthetic fault-injection corpus
uv run python -m cascaid.train                # train GNN vs XGBoost baseline, save pretrained artifact
```

Or, once `uv sync` has installed the project in editable mode, use the unified
CLI directly: `uv run cascaid demo`, `uv run cascaid serve ...`, etc.

## Test pyramid

```
uv run pytest tests/unit -q          # fast, no I/O
uv run pytest tests/integration -q   # real demo pipeline -> snapshot -> labeling wiring
uv run pytest tests/e2e -q           # actual CLI entry points end-to-end
```

## Branching & CI

- `master` -- production. `staging` -- pre-prod integration. `feature/*` -- work in progress, PR into `staging`, then `staging` -> `master`.
- Pushing a `feature/**` branch runs lint + unit tests (fast feedback).
- A PR into `staging` or `master` runs the full pyramid: lint, unit, integration, e2e (`.github/workflows/ci.yml`).
- A pushed tag `vX.Y.Z` retrains the model on the full synthetic corpus and publishes `models/pretrained_base.pt` to a GitHub Release (`.github/workflows/release.yml`).

Branch protection (require PR + passing checks into `staging`/`master`) is one-time setup a human has to do with an authenticated GitHub session -- run:

```
bash scripts/setup-branch-protection.sh
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
