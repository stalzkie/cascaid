# Shared image for every Python-side service (Model Serving, Dashboard API, the
# demo seed step) -- docker-compose selects which one runs via `command:`.
# PRD 4.4: one artifact, not five services the customer has to wire up by hand.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first so the (slow -- torch/torch-geometric) install layer is
# cached across rebuilds that only touch application code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
