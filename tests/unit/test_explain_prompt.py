"""Unit seam: cascaid.explain.prompt's pure helpers (PRD 7 LLM-generated risk
explanations). No LLM call involved -- see test_explain_client.py under
tests/integration for the network-facing half."""

from cascaid.explain.prompt import build_explanation_prompt, callees_of, node_feature_dict


class _FakeData:
    def __init__(self, node_order, x):
        self.node_order = node_order
        self.x = x


def test_node_feature_dict_reads_the_first_num_features_columns_in_order():
    # NUM_FEATURES=4 -> [latency_ms, error_rate, retry_rate, token_cost], followed
    # by node-type one-hot columns this helper must ignore.
    x = [
        [10.0, 0.1, 0.2, 0.3, 1, 0, 0, 0],
        [400.0, 0.9, 0.5, 0.05, 0, 0, 1, 0],
    ]
    data = _FakeData(node_order=["agent", "vector_store"], x=x)

    assert node_feature_dict(data, "vector_store") == {
        "latency_ms": 400.0,
        "error_rate": 0.9,
        "retry_rate": 0.5,
        "token_cost": 0.05,
    }


def test_callees_of_returns_only_nodes_this_node_calls():
    edges = [("agent", "vector_store"), ("agent", "primary_model"), ("vector_store", "primary_model")]

    assert callees_of("agent", edges) == ["vector_store", "primary_model"]
    assert callees_of("primary_model", edges) == []


def test_build_explanation_prompt_includes_the_node_score_and_neighbor_facts():
    prompt = build_explanation_prompt(
        node_name="research_agent",
        node_type="agent",
        risk_score=0.87,
        own_features={"latency_ms": 120.0, "error_rate": 0.05, "retry_rate": 0.0, "token_cost": 0.02},
        neighbor_features={
            "vector_store": {
                "node_type": "vector_store",
                "latency_ms": 900.0,
                "error_rate": 0.4,
                "retry_rate": 0.1,
                "token_cost": 0.0,
            }
        },
    )

    assert "research_agent" in prompt
    assert "0.87" in prompt
    assert "vector_store" in prompt
    assert "900.0" in prompt
