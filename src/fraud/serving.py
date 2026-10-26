"""Load the serving bundle and score one transaction from its card history."""

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud import features, policy

ROOT = Path(__file__).resolve().parents[2]
ART = Path(os.getenv("FRAUD_ARTIFACTS", ROOT / "artifacts" / "final"))


def build_one(history: pd.DataFrame, tx: dict) -> dict:
    """Same features as features.build, for one row and its prior rows only."""
    ts = pd.Timestamp(tx["ts"])
    amt = float(tx["amt"])
    n = len(history)
    prior_mean = float(history.amt.mean()) if n else np.nan
    same = history[history.merchant == tx["merchant"]]
    window = lambda s: int((history.ts >= ts - pd.Timedelta(seconds=s)).sum())  # noqa: E731
    return {
        "amt": amt,
        "amt_log": float(np.log1p(amt)),
        "hour": ts.hour,
        "weekday": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
        "is_night": int(ts.hour >= 22 or ts.hour <= 5),
        "city_pop": float(tx["city_pop"]),
        "tx_count_prior": n,
        "amt_prior_mean": prior_mean,
        "amt_ratio_mean": amt / (prior_mean + 1.0) if n else 1.0,
        "secs_since_prev": (ts - history.ts.max()).total_seconds() if n else -1.0,
        "secs_since_same_merchant": (ts - same.ts.max()).total_seconds() if len(same) else -1.0,
        "card_age_days": (ts - history.ts.min()).total_seconds() / 86400 if n else 0.0,
        "new_merchant": int(tx["merchant"] not in set(history.merchant)),
        "new_category": int(tx["category"] not in set(history.category)),
        "tx_count_1h": window(3600),
        "tx_count_24h": window(86400),
        "tx_count_7d": window(604800),
        "category": tx["category"],
        "merchant": tx["merchant"],
        "state": tx["state"],
    }


class Bundle:
    def __init__(self, path: Path = ART):
        self.card = json.loads((path / "model_card.json").read_text(encoding="utf-8"))
        self.model = lgb.Booster(model_file=str(path / "model.txt"))
        self.cal = joblib.load(path / "calibrator.joblib")
        self.threshold = float(self.card["threshold"])
        self.names = features.NUMERIC + features.CATEGORICAL
        self.cats = self.model.pandas_categorical

    def frame(self, row: dict) -> pd.DataFrame:
        x = pd.DataFrame([row])[self.names]
        for col, cats in zip(features.CATEGORICAL, self.cats):
            x[col] = pd.Categorical(x[col].astype(str), categories=[str(c) for c in cats])
        return x

    def score(self, row: dict) -> dict:
        x = self.frame(row)
        raw = float(self.model.predict(x)[0])
        p = float(policy.apply_platt(self.cal, np.array([raw]))[0])
        contrib = self.model.predict(x, pred_contrib=True)[0][:-1]
        top = sorted(zip(self.names, contrib), key=lambda t: -abs(t[1]))[:5]
        return {
            "score": p,
            "raw_score": raw,
            "decision": "flag" if p >= self.threshold else "pass",
            "threshold": self.threshold,
            "top_drivers": [{"feature": f, "contribution": float(v)} for f, v in top],
        }
