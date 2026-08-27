import numpy as np

from cascaid.metrics import RunTrace, lead_time_accuracy, pr_auc


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
