import json

from cascaid.ingestion.langfuse_import import parse_langfuse_scores, scores_below_threshold


def _score(name="helpfulness", value=0.9, data_type="NUMERIC", timestamp="2026-08-01T00:00:00Z"):
    return {"id": "s1", "name": name, "value": value, "dataType": data_type, "timestamp": timestamp}


def test_parse_langfuse_scores_reads_a_bare_array(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps([_score(), _score(name="toxicity")]), encoding="utf-8")

    scores = parse_langfuse_scores(path)

    assert len(scores) == 2
    assert scores[0]["name"] == "helpfulness"


def test_parse_langfuse_scores_unwraps_a_data_envelope(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"data": [_score()], "meta": {"limit": 50}}), encoding="utf-8")

    scores = parse_langfuse_scores(path)

    assert len(scores) == 1


def test_scores_below_threshold_keeps_only_low_numeric_scores():
    scores = [
        _score(name="ok", value=0.9),
        _score(name="bad", value=0.2),
        _score(name="boolean-ignored", value=True, data_type="BOOLEAN"),
        _score(name="no-value", value=None),
    ]

    degraded = scores_below_threshold(scores, threshold=0.5)

    assert [s["name"] for s in degraded] == ["bad"]
