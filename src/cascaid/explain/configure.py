"""Opt-in switch for LLM-generated risk explanations (PRD 7), off by default --
mirrors cascaid.alerting.configure's shape. Bring-your-own-endpoint: --base-url
can point at OpenAI, an OpenAI-compatible proxy, or a self-hosted vLLM/Ollama
instance, so a customer who doesn't want pipeline data leaving their cluster
can point this at their own model instead (see
docs/Client_Readiness_and_YC_Grade_Assessment.md section 4).

    python -m cascaid.explain.configure --database-url ... --enable \
        --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o-mini
    python -m cascaid.explain.configure --database-url ... --disable

--api-key is encrypted at rest (ADR 0005, see cascaid.storage.secrets) -- setting it
requires CASCAID_CONFIG_ENCRYPTION_KEY to already be set in this process's environment.
"""

from __future__ import annotations

import argparse

from cascaid.storage.db import make_session_factory
from cascaid.storage.repository import set_config
from cascaid.storage.secrets import set_secret_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", action="store_true")
    group.add_argument("--disable", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    session_factory = make_session_factory(args.database_url)
    with session_factory() as session:
        if args.enable:
            set_config(session, "llm_explanations_enabled", "true")
        if args.disable:
            set_config(session, "llm_explanations_enabled", "false")
        if args.base_url is not None:
            set_config(session, "llm_base_url", args.base_url)
        if args.api_key is not None:
            set_secret_config(session, "llm_api_key", args.api_key)
        if args.model is not None:
            set_config(session, "llm_model", args.model)


if __name__ == "__main__":
    main()
