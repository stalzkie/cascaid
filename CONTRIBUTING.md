# Contributing to Cascaid

Thanks for considering a contribution. This doc covers the local dev loop,
what CI checks, and how PRs get merged.

## Ways to contribute

- **Bug reports** and **feature requests**: open a [GitHub issue](https://github.com/stalzkie/cascaid/issues).
- **Docs**: the `docs/` folder is an Obsidian vault covering the product spec,
  design research, and accuracy logs -- see `docs/README.md` for the index.
- **Code**: see the workflow below.

Before starting significant work (a new instrumentation target, a new model
architecture, a dashboard redesign), open an issue first so the approach can
be discussed -- it's easier to agree on a direction before code exists than
to rework a finished PR.

## Local setup

Backend (Python 3.12, managed with [uv](https://docs.astral.sh/uv/)):

```
uv sync --group dev
```

Frontend (Node 20):

```
cd frontend
npm install
```

Full stack, if you need Postgres + the dashboard + model server running
locally:

```
cp .env.example .env   # then change POSTGRES_PASSWORD
docker compose up
```

## Before you open a PR

Run the same checks CI runs, scoped to what you touched:

```
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit -q          # fast, no I/O -- run this always
uv run pytest tests/integration -q   # needs CASCAID_TEST_DATABASE_URL, see below
uv run pytest tests/e2e -q           # exercises the actual CLI entry points
```

```
cd frontend
npm run typecheck
npm test
npm run build
```

Integration tests need a real Postgres reachable via
`CASCAID_TEST_DATABASE_URL` (see `.github/workflows/ci.yml` for how CI wires
one up with `docker`, or point it at the `docker compose up` Postgres). If you
can't run integration/e2e locally, say so in the PR description and CI will
still run them on your PR.

## PR workflow

- Branch from `staging`, not `master`.
- Target your PR at `staging`. `staging -> master` promotions are done
  separately once a batch of work has soaked.
- CI runs lint, unit, integration, e2e, the frontend suite, and a Docker
  Compose smoke test on every PR into `staging`/`master`
  (`.github/workflows/ci.yml`) -- all of it needs to be green before merge.
- Keep PRs scoped to one change. Large, unrelated changes bundled together are
  harder to review and to revert if something's wrong.
- A maintainer reviews and merges every PR. Branch protection doesn't force a
  second approval (the repo also carries agent-authored and maintainer's own
  PRs merged solo), but that's a policy for trusted commits, not a signal
  that outside contributions skip review -- expect a look before merge.

## Code style

- Python: [ruff](https://docs.astral.sh/ruff/) for both linting and
  formatting (`select = ["E", "F", "I"]`, 120-col lines) -- config lives in
  `pyproject.toml`.
- Tests are split by cost/scope: `tests/unit` (fast, no I/O),
  `tests/integration` (multiple modules wired together against a real
  Postgres), `tests/e2e` (actual CLI entry points). Put new tests in the
  cheapest tier that still exercises the behavior honestly.
- No enforced frontend formatter beyond `tsc --noEmit`; match the existing
  style in the file you're editing.

## Reporting security issues

Do not open a public issue for a security vulnerability -- see
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree your contributions are licensed under the
project's [Apache 2.0 license](LICENSE).
