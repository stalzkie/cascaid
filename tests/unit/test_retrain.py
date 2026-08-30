"""Unit seam: cascaid.retrain.should_swap_model -- the gate deciding whether a
newly retrained model replaces the one currently served (PRD 4.3 fine-tuning,
kept safe per docs/Real_Data_Retraining_Plan.md: a small noisy batch of real
incidents must not be able to silently make the served model worse)."""

import math

from cascaid.retrain import should_swap_model


def test_swaps_when_pr_auc_clears_the_floor():
    assert should_swap_model(0.85, min_pr_auc=0.7) is True


def test_does_not_swap_when_pr_auc_is_below_the_floor():
    assert should_swap_model(0.5, min_pr_auc=0.7) is False


def test_does_not_swap_when_pr_auc_equals_the_floor():
    assert should_swap_model(0.7, min_pr_auc=0.7) is True


def test_does_not_swap_on_nan_pr_auc():
    assert should_swap_model(math.nan, min_pr_auc=0.0) is False
