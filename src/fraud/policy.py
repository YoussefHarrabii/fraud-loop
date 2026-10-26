"""Decision policy: calibrate a score, pick a recall-target threshold, decide."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression().fit(logit(p).reshape(-1, 1), y)


def apply_platt(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return cal.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def recall_threshold(p_val: np.ndarray, y_val: np.ndarray, target: float) -> float:
    return float(np.quantile(p_val[y_val == 1], 1 - target))


def decide(p: np.ndarray, threshold: float) -> np.ndarray:
    return p >= threshold


def row_cost(y, p, amt, threshold, false_alarm_cost=7.0, recovery=0.8):
    y = np.asarray(y)
    p = np.asarray(p)
    amt = np.asarray(amt, dtype=float)
    flag = p >= threshold
    return np.where(
        flag,
        np.where(y == 1, 0.0, false_alarm_cost),
        np.where(y == 1, amt * recovery, 0.0),
    )


def bootstrap_cost_diff(y, p_cand, p_prod, amt, th_cand, th_prod,
                        false_alarm_cost=7.0, recovery=0.8, n=1000, seed=42) -> dict:
    diff = (
        row_cost(y, p_cand, amt, th_cand, false_alarm_cost, recovery)
        - row_cost(y, p_prod, amt, th_prod, false_alarm_cost, recovery)
    )
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {
        "mean": float(diff.mean()),
        "low": float(np.percentile(boots, 2.5)),
        "high": float(np.percentile(boots, 97.5)),
    }


def should_promote(ci_high: float, cand_recall: float, prod_recall: float, margin: float = 0.02) -> bool:
    return bool(ci_high < 0 and cand_recall >= prod_recall - margin)
