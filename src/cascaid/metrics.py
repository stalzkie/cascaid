"""Metrics discipline required before the graph model is justified (PRD 6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


@dataclass
class RunTrace:
    run_id: str
    fault_onset_step: int
    cascade_step: int
    steps: list[int]
    scores: list[float]  # risk score for the node(s) of interest at each step


def lead_time_accuracy(traces: list[RunTrace], threshold: float) -> dict:
    """For each faulty run, find the first step >= fault_onset_step where the
    risk score crosses `threshold`, and how many steps that is ahead of
    cascade_step (positive = caught before full manifestation)."""
    lead_times = []
    detected = 0
    for tr in traces:
        crossed_step = None
        for step, score in zip(tr.steps, tr.scores):
            if step >= tr.fault_onset_step and score >= threshold:
                crossed_step = step
                break
        if crossed_step is not None:
            detected += 1
            lead_times.append(tr.cascade_step - crossed_step)
    return {
        "num_runs": len(traces),
        "detected": detected,
        "detection_rate": detected / len(traces) if traces else float("nan"),
        "mean_lead_time_steps": float(np.mean(lead_times)) if lead_times else float("nan"),
    }
