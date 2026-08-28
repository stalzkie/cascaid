from cascaid.train import build_traces


def test_build_traces_uses_max_score_across_multiple_epicenters():
    # compound_cascade has two epicenters (primary_model, vector_store) --
    # the trace used for lead-time detection should reflect whichever
    # epicenter the model actually flagged, not just the first one.
    manifest = [
        {"run_id": "r1", "scenario": "compound_cascade", "fault_onset_step": 5, "cascade_step": 15},
    ]
    node_scores = {
        "r1": {
            5: {"primary_model": 0.9, "vector_store": 0.1},
            6: {"primary_model": 0.2, "vector_store": 0.8},
        },
    }

    traces = build_traces(manifest, {"r1"}, node_scores)

    assert len(traces) == 1
    assert traces[0].scores == [0.9, 0.8]


def test_build_traces_still_works_for_single_epicenter_scenarios():
    manifest = [
        {"run_id": "r1", "scenario": "rate_limit_model", "fault_onset_step": 5, "cascade_step": 15},
    ]
    node_scores = {"r1": {5: {"primary_model": 0.7, "vector_store": 0.1}}}

    traces = build_traces(manifest, {"r1"}, node_scores)

    assert traces[0].scores == [0.7]


def test_build_traces_skips_baseline_runs():
    manifest = [{"run_id": "r1", "scenario": "baseline", "fault_onset_step": None, "cascade_step": None}]

    traces = build_traces(manifest, {"r1"}, {"r1": {}})

    assert traces == []
