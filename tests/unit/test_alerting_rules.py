from cascaid.alerting.rules import evaluate_risk


def test_evaluate_risk_returns_no_alerts_below_threshold():
    alerts = evaluate_risk(
        run_id="run-1",
        node_scores={"agent": 0.1, "store": 0.3},
        node_types={"agent": "agent", "store": "vector_store"},
        threshold=0.5,
    )
    assert alerts == []


def test_evaluate_risk_flags_nodes_at_or_above_threshold():
    alerts = evaluate_risk(
        run_id="run-1",
        node_scores={"agent": 0.1, "store": 0.9},
        node_types={"agent": "agent", "store": "vector_store"},
        threshold=0.5,
    )
    assert len(alerts) == 1
    assert alerts[0].node_name == "store"
    assert alerts[0].risk_score == 0.9
    assert alerts[0].run_id == "run-1"


def test_evaluate_risk_message_names_vector_store_and_quality_impact():
    alerts = evaluate_risk(
        run_id="run-1",
        node_scores={"retriever": 0.87},
        node_types={"retriever": "vector_store"},
        threshold=0.5,
    )
    assert "retriever" in alerts[0].message
    assert "0.87" in alerts[0].message
    assert "generation quality" in alerts[0].message


def test_evaluate_risk_message_names_model_endpoint_and_fallback_impact():
    alerts = evaluate_risk(
        run_id="run-1",
        node_scores={"primary_model": 0.95},
        node_types={"primary_model": "model_endpoint"},
        threshold=0.5,
    )
    assert "primary_model" in alerts[0].message
    assert "fallback" in alerts[0].message or "latency" in alerts[0].message
