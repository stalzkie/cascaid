"""Fault scenario definitions (PRD Phase 0 step 2 / Section 6.1):
rate-limiting a model endpoint, degrading a vector DB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCENARIOS = ["baseline", "rate_limit_model", "vector_db_degradation"]


@dataclass
class ScenarioConfig:
    name: str
    total_steps: int
    fault_onset_step: int | None
    ramp_steps: int = 10

    @property
    def cascade_step(self) -> int | None:
        if self.fault_onset_step is None:
            return None
        return self.fault_onset_step + self.ramp_steps

    def fault_progress(self, step: int) -> float:
        """0.0 before fault_onset_step, ramps linearly to 1.0 by cascade_step, stays 1.0 after."""
        if self.fault_onset_step is None or step < self.fault_onset_step:
            return 0.0
        return min(1.0, (step - self.fault_onset_step) / self.ramp_steps)


def make_scenario(name: str, total_steps: int, rng: np.random.Generator, ramp_steps: int = 10) -> ScenarioConfig:
    if name == "baseline":
        return ScenarioConfig(name=name, total_steps=total_steps, fault_onset_step=None, ramp_steps=ramp_steps)
    lo = int(total_steps * 0.35)
    hi = int(total_steps * 0.55)
    onset = int(rng.integers(lo, hi))
    return ScenarioConfig(name=name, total_steps=total_steps, fault_onset_step=onset, ramp_steps=ramp_steps)
