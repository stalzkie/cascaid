from __future__ import annotations

import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig


class VectorStore:
    name = "vector_store"

    def query(self, step: int, scenario: ScenarioConfig, rng: np.random.Generator) -> dict:
        base_latency = max(1.0, float(rng.normal(60, 8)))
        if scenario.name in ("vector_db_degradation", "compound_cascade"):
            progress = scenario.fault_progress(step)
            base_latency += progress * max(0.0, float(rng.normal(700, 40)))

        # Flaky vector store: intermittent errors without a latency signature,
        # unlike vector_db_degradation's smooth latency ramp -- a different
        # manifestation of the same epicenter so the model can't just key off
        # "latency went up" to recognize vector_store risk.
        error_p = 0.02
        if scenario.name == "vector_store_flaky":
            progress = scenario.fault_progress(step)
            error_p = 0.02 + progress * 0.5

        error = bool(rng.random() < error_p)
        return {
            "latency_ms": base_latency,
            "error": error,
            "retried": False,
            "token_cost": 0.0,
        }
