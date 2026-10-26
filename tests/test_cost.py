import numpy as np

from fraud import policy


def test_row_cost_miss_and_false_alarm():
    y = np.array([1, 0, 0])
    p = np.array([0.1, 0.9, 0.1])
    amt = np.array([100.0, 50.0, 50.0])
    cost = policy.row_cost(y, p, amt, 0.5, false_alarm_cost=7.0, recovery=0.8)
    assert cost.tolist() == [80.0, 7.0, 0.0]


def test_threshold_is_inclusive():
    y = np.array([1])
    p = np.array([0.5])
    cost = policy.row_cost(y, p, np.array([100.0]), 0.5)
    assert cost.tolist() == [0.0]


def test_bootstrap_identical_models_is_zero():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    amt = np.array([100.0, 20.0, 50.0, 10.0])
    ci = policy.bootstrap_cost_diff(y, p, p, amt, 0.5, 0.5)
    assert ci["mean"] == 0.0
    assert ci["low"] == 0.0
    assert ci["high"] == 0.0
    assert not policy.should_promote(ci["high"], 1.0, 1.0)


def test_bootstrap_candidate_better():
    y = np.array([1, 0])
    amt = np.array([100.0, 100.0])
    ci = policy.bootstrap_cost_diff(y, np.array([0.9, 0.1]), np.array([0.1, 0.9]), amt, 0.5, 0.5)
    assert ci["mean"] < 0
    assert ci["high"] < 0
    assert policy.should_promote(ci["high"], 1.0, 0.0)


def test_recall_guard_blocks_cheaper_model():
    assert not policy.should_promote(-1.0, 0.85, 0.90, margin=0.02)
    assert policy.should_promote(-1.0, 0.89, 0.90, margin=0.02)
