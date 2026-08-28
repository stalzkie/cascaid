import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig
from cascaid_demo.mock_llm_gateway import ModelGateway

N = 500


def _mean_primary_cost_and_error(scenario: ScenarioConfig, step: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    gateway = ModelGateway()
    costs, errors = [], []
    for _ in range(N):
        events, _used = gateway.call(step, scenario, rng)
        primary_ev = events[0][1]
        costs.append(primary_ev["token_cost"])
        errors.append(1.0 if primary_ev["error"] else 0.0)
    return float(np.mean(costs)), float(np.mean(errors))


def test_cost_spike_model_elevates_cost_without_elevating_error_rate():
    baseline = ScenarioConfig(name="baseline", total_steps=10, fault_onset_step=None)
    at_fault = ScenarioConfig(name="cost_spike_model", total_steps=10, fault_onset_step=0, ramp_steps=1)

    baseline_cost, baseline_error = _mean_primary_cost_and_error(baseline, step=5)
    fault_cost, fault_error = _mean_primary_cost_and_error(at_fault, step=1)

    assert fault_cost > baseline_cost * 3
    assert fault_error < baseline_error + 0.05


def test_rate_limit_model_still_elevates_error_as_before():
    at_fault = ScenarioConfig(name="rate_limit_model", total_steps=10, fault_onset_step=0, ramp_steps=1)

    _cost, error = _mean_primary_cost_and_error(at_fault, step=1)

    assert error > 0.5
