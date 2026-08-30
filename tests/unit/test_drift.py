import numpy as np

from cascaid.serving.drift import compute_drift, compute_reference, load_reference, save_reference


def test_compute_drift_is_near_zero_when_observed_matches_reference():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(1000, 2))
    reference = compute_reference(features, feature_names=["a", "b"], n_bins=10)

    observed = rng.uniform(0, 1, size=(200, 2))
    scores = compute_drift(reference, observed, feature_names=["a", "b"])

    assert scores["a"] < 0.1
    assert scores["b"] < 0.1


def test_compute_drift_is_large_when_observed_distribution_shifts():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(1000, 1))
    reference = compute_reference(features, feature_names=["a"], n_bins=10)

    shifted = rng.uniform(5, 6, size=(200, 1))
    scores = compute_drift(reference, shifted, feature_names=["a"])

    assert scores["a"] > 1.0


def test_compute_reference_handles_a_zero_variance_feature():
    # Real topologies can have a feature that's constant across the whole training
    # set (e.g. every node starts with retry_rate=0) -- quantile edges degenerate
    # to a single point, which must not raise or produce an unusable reference.
    features = np.zeros((100, 1))

    reference = compute_reference(features, feature_names=["constant"], n_bins=10)

    assert len(reference["constant"]["bin_edges"]) >= 2
    observed = np.zeros((20, 1))
    scores = compute_drift(reference, observed, feature_names=["constant"])
    assert scores["constant"] < 0.1


def test_reference_round_trips_through_json(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(100, 1))
    reference = compute_reference(features, feature_names=["a"], n_bins=5)
    path = tmp_path / "reference.json"

    save_reference(reference, path)
    loaded = load_reference(path)

    assert loaded == reference
