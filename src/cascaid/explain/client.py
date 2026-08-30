"""LLM call for a risk explanation (PRD 7), against any OpenAI-compatible
/chat/completions endpoint -- OpenAI/Anthropic-compatible proxies, or a
self-hosted vLLM/Ollama instance for customers who don't want pipeline data
leaving their cluster (opt-in, bring-your-own-endpoint -- see
docs/Client_Readiness_and_YC_Grade_Assessment.md section 4). Never raises: a
down or misconfigured LLM endpoint must not break risk serving, the same
principle cascaid.alerting.dispatch.send_webhook follows."""

from __future__ import annotations

import httpx


def generate_explanation(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 15.0,
) -> str | None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = httpx.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None
