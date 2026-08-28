import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig
from cascaid_demo.mock_vector_db import VectorStore

N = 500


def _mean_latency_and_error(scenario: ScenarioConfig, step: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    store = VectorStore()
    latencies, errors = [], []
    for _ in range(N):
        ev = store.query(step, scenario, rng)
        latencies.append(ev["latency_ms"])
        errors.append(1.0 if ev["error"] else 0.0)
    return float(np.mean(latencies)), float(np.mean(errors))


def test_vector_store_flaky_elevates_error_without_elevating_latency():
    baseline = ScenarioConfig(name="baseline", total_steps=10, fault_onset_step=None)
    at_fault = ScenarioConfig(name="vector_store_flaky", total_steps=10, fault_onset_step=0, ramp_steps=1)

    baseline_latency, baseline_error = _mean_latency_and_error(baseline, step=5)
    fault_latency, fault_error = _mean_latency_and_error(at_fault, step=1)

    assert fault_error > baseline_error + 0.3
    assert fault_latency < baseline_latency + 20


def test_vector_db_degradation_still_elevates_latency_as_before():
    at_fault = ScenarioConfig(name="vector_db_degradation", total_steps=10, fault_onset_step=0, ramp_steps=1)

    latency, _error = _mean_latency_and_error(at_fault, step=1)

    assert latency > 300


def test_compound_cascade_elevates_vector_store_latency_too():
    at_fault = ScenarioConfig(name="compound_cascade", total_steps=10, fault_onset_step=0, ramp_steps=1)

    latency, _error = _mean_latency_and_error(at_fault, step=1)

    assert latency > 300
