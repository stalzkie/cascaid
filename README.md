# Cascaid

Predictive cascading-failure intelligence for AI-native systems. See `docs/Cascaid_PRD (1).md` for the full product spec.

## Development

```
uv sync                              # install pinned Python 3.12 env + deps
uv run python -m cascaid_demo.run_scenarios   # generate synthetic fault-injection corpus
uv run python -m cascaid.train                # train GNN vs XGBoost baseline, save pretrained artifact
```

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
