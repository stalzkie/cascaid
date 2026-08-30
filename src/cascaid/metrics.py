"""Metrics discipline required before the graph model is justified (PRD 6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean squared error between predicted probability and true label.

    PR-AUC (above) only measures ranking quality -- it can't tell a
    well-calibrated 0.8 from a 0.8 that's really a 0.3 in disguise. Brier
    score is the standard proper scoring rule for "does the number mean
    what it says" (PRD 1.1's "calibrated probability score")."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Bins predictions by score into `n_bins` equal-width buckets and
    averages, per bucket, the gap between mean predicted confidence and
    mean empirical accuracy -- weighted by bucket size. 0.0 is perfectly
    calibrated; complements brier_score with a human-readable "how far off"
    number instead of a squared-error scale."""
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_score, bin_edges[1:-1], right=True), 0, n_bins - 1)
    total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        mean_confidence = float(np.mean(y_score[mask]))
        mean_accuracy = float(np.mean(y_true[mask]))
        ece += (mask.sum() / total) * abs(mean_confidence - mean_accuracy)
    return float(ece)


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
