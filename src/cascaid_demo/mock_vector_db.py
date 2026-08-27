from __future__ import annotations

import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig


class VectorStore:
    name = "vector_store"

    def query(self, step: int, scenario: ScenarioConfig, rng: np.random.Generator) -> dict:
        base_latency = max(1.0, float(rng.normal(60, 8)))
        if scenario.name == "vector_db_degradation":
            progress = scenario.fault_progress(step)
            base_latency += progress * max(0.0, float(rng.normal(700, 40)))
        error = bool(rng.random() < 0.02)
        return {
            "latency_ms": base_latency,
            "error": error,
            "retried": False,
            "token_cost": 0.0,
        }
