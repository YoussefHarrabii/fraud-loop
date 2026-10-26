import numpy as np

from fraud import policy


def test_logit_roundtrip():
    p = np.array([0.01, 0.5, 0.99])
    assert np.allclose(1 / (1 + np.exp(-policy.logit(p))), p, atol=1e-6)


def test_platt_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(400)
    y = (rng.random(400) < 0.2).astype(int)
    cal = policy.fit_platt(p, y)
    q = policy.apply_platt(cal, p)
    assert q.min() >= 0 and q.max() <= 1
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_recall_threshold_meets_target():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = (rng.random(1000) < 0.2).astype(int)
    threshold = policy.recall_threshold(p, y, 0.9)
    flag = policy.decide(p, threshold)
    # Quantile interpolation at a tie can land a fraction under the target.
    assert 0.89 <= flag[y == 1].mean() <= 0.95
