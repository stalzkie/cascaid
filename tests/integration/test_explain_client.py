"""Integration seam: cascaid.explain.client.generate_explanation against a real
local HTTP server standing in for an OpenAI-compatible endpoint (OpenAI itself,
or a self-hosted vLLM/Ollama instance -- PRD 7 LLM-generated risk explanations,
opt-in bring-your-own-endpoint per docs/Client_Readiness_and_YC_Grade_Assessment.md
section 4)."""

import pytest

from cascaid.explain.client import generate_explanation


@pytest.mark.integration
def test_generate_explanation_returns_the_completion_text(httpserver):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {"choices": [{"message": {"content": "The vector store's latency spiked."}}]}
    )

    result = generate_explanation(
        "why is this risky?",
        base_url=httpserver.url_for("/v1"),
        api_key="sk-test",
        model="gpt-4o-mini",
    )

    assert result == "The vector store's latency spiked."


@pytest.mark.integration
def test_generate_explanation_sends_the_prompt_and_model_and_auth_header(httpserver):
    httpserver.expect_request(
        "/v1/chat/completions",
        method="POST",
        headers={"Authorization": "Bearer sk-test"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "why is this risky?"}]},
    ).respond_with_json({"choices": [{"message": {"content": "ok"}}]})

    result = generate_explanation(
        "why is this risky?",
        base_url=httpserver.url_for("/v1"),
        api_key="sk-test",
        model="gpt-4o-mini",
    )

    assert result == "ok"


@pytest.mark.integration
def test_generate_explanation_returns_none_on_server_error(httpserver):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_data(status=500)

    result = generate_explanation("why?", base_url=httpserver.url_for("/v1"), api_key="sk-test", model="gpt-4o-mini")

    assert result is None


@pytest.mark.integration
def test_generate_explanation_returns_none_when_unreachable():
    result = generate_explanation(
        "why?", base_url="http://127.0.0.1:1/v1", api_key="sk-test", model="gpt-4o-mini", timeout=0.5
    )

    assert result is None


@pytest.mark.integration
def test_generate_explanation_returns_none_on_unexpected_response_shape(httpserver):
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json({"unexpected": "shape"})

    result = generate_explanation("why?", base_url=httpserver.url_for("/v1"), api_key="sk-test", model="gpt-4o-mini")

    assert result is None
