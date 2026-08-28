"""Simulates LiteLLM-style model routing: primary endpoint, fallback on error/rate-limit."""

from __future__ import annotations

import numpy as np

from cascaid_demo.fault_injection import ScenarioConfig


class ModelGateway:
    primary = "primary_model"
    fallback = "fallback_model"

    def call(self, step: int, scenario: ScenarioConfig, rng: np.random.Generator) -> tuple[list[tuple[str, dict]], str]:
        primary_error_p = 0.03
        if scenario.name in ("rate_limit_model", "compound_cascade"):
            progress = scenario.fault_progress(step)
            primary_error_p = 0.03 + progress * 0.85

        # Cost spike: a silent fallback to a slower/pricier model (PRD 5.2
        # Model Serving edge features) -- cost and latency rise while the
        # error rate stays normal, unlike rate_limit_model's failure-driven
        # signature. Distinguishing these two teaches the model that risk
        # isn't just "elevated errors".
        cost_mean, latency_mean = 0.02, 180.0
        if scenario.name == "cost_spike_model":
            progress = scenario.fault_progress(step)
            cost_mean = 0.02 + progress * 0.13
            latency_mean = 180.0 + progress * 100.0

        primary_failed = bool(rng.random() < primary_error_p)
        primary_latency = max(1.0, float(rng.normal(latency_mean, 20)))
        events = [
            (
                self.primary,
                {
                    "latency_ms": primary_latency,
                    "error": primary_failed,
                    "retried": False,
                    "token_cost": max(0.0, float(rng.normal(cost_mean, 0.003))),
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
