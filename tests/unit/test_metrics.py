import numpy as np
import pytest

from cascaid.metrics import RunTrace, brier_score, expected_calibration_error, lead_time_accuracy, pr_auc


def test_pr_auc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == 1.0


def test_pr_auc_all_same_class_is_nan():
    assert np.isnan(pr_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))


def test_lead_time_accuracy_detects_before_cascade():
    trace = RunTrace(
        run_id="r1",
        fault_onset_step=10,
        cascade_step=20,
        steps=list(range(0, 25)),
        scores=[0.1] * 10 + [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9] + [0.9] * 5,
    )
    result = lead_time_accuracy([trace], threshold=0.5)
    assert result["detected"] == 1
    assert result["detection_rate"] == 1.0
    # score crosses 0.5 at step 13 (10 + index 3), cascade_step=20 -> lead = 7
    assert result["mean_lead_time_steps"] == 7


def test_lead_time_accuracy_missed_detection():
    trace = RunTrace(
        run_id="r2",
        fault_onset_step=10,
        cascade_step=20,
        steps=list(range(0, 20)),
        scores=[0.1] * 20,
    )
    result = lead_time_accuracy([trace], threshold=0.5)
    assert result["detected"] == 0
    assert result["detection_rate"] == 0.0


def test_brier_score_perfect_predictions_is_zero():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_score) == 0.0


def test_brier_score_worked_example():
    # (0.1-0)^2 + (0.9-0)^2 + (0.9-1)^2 + (0.1-1)^2 = 0.01+0.81+0.01+0.81 = 1.64; /4 = 0.41
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.9, 0.9, 0.1])
    assert brier_score(y_true, y_score) == pytest.approx(0.41)


def test_brier_score_empty_input_is_nan():
    assert np.isnan(brier_score(np.array([]), np.array([])))


def test_expected_calibration_error_worked_example():
    # bin [0, 0.5): scores 0.1, 0.2 -> true 0, 0 -> |mean_conf 0.15 - mean_acc 0.0| = 0.15, weight 0.5
    # bin [0.5, 1.0]: scores 0.8, 0.9 -> true 1, 1 -> |mean_conf 0.85 - mean_acc 1.0| = 0.15, weight 0.5
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y_true, y_score, n_bins=2) == pytest.approx(0.15)


def test_expected_calibration_error_perfect_calibration_is_zero():
    # Every score exactly matches the empirical rate within its bin.
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(y_true, y_score, n_bins=2) == 0.0


def test_expected_calibration_error_empty_input_is_nan():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([]), n_bins=10))
