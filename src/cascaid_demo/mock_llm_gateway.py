"""Simulates LiteLLM-style model routing: primary endpoint, fallback on error/rate-limit."""

from __future__ import annotations

import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig


class ModelGateway:
    primary = "primary_model"
    fallback = "fallback_model"

    def call(self, step: int, scenario: ScenarioConfig, rng: np.random.Generator) -> tuple[list[tuple[str, dict]], str]:
        primary_error_p = 0.03
        if scenario.name == "rate_limit_model":
            progress = scenario.fault_progress(step)
            primary_error_p = 0.03 + progress * 0.85

        primary_failed = bool(rng.random() < primary_error_p)
        primary_latency = max(1.0, float(rng.normal(180, 20)))
        events = [
            (
                self.primary,
                {
                    "latency_ms": primary_latency,
                    "error": primary_failed,
                    "retried": False,
                    "token_cost": max(0.0, float(rng.normal(0.02, 0.003))),
                },
            )
        ]
        used = self.primary
        if primary_failed:
            fallback_latency = max(1.0, float(rng.normal(260, 30)))
            events.append(
                (
                    self.fallback,
                    {
                        "latency_ms": fallback_latency,
                        "error": bool(rng.random() < 0.02),
                        "retried": True,
                        "token_cost": max(0.0, float(rng.normal(0.05, 0.005))),
                    },
                )
            )
            used = self.fallback
        return events, used
