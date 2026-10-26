"""Drift and live-operations metrics from the request log and analyst decisions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "artifacts" / "final" / "reference.json"

DRIFT_FEATURES = [
    "amt", "hour", "tx_count_1h", "tx_count_24h", "tx_count_7d",
    "amt_ratio_mean", "secs_since_prev", "city_pop",
]
STABLE, MODERATE = 0.10, 0.25


def psi(expected, actual, bins: int = 10) -> float | None:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) < 20 or len(a) < 20:
        return None
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    a = np.clip(a, edges[0], edges[-1])
    e = np.clip(e, edges[0], edges[-1])
    eh, _ = np.histogram(e, bins=edges)
    ah, _ = np.histogram(a, bins=edges)
    ep = np.clip(eh / len(e), 1e-6, None)
    ap = np.clip(ah / len(a), 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def ks(expected, actual) -> float | None:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) < 20 or len(a) < 20:
        return None
    return float(ks_2samp(e, a).statistic)


def save_profile(df: pd.DataFrame) -> None:
    out = {"n": int(len(df)), "features": {}}
    for col in DRIFT_FEATURES:
        v = df[col].dropna().to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        sample = rng.choice(v, size=min(5000, len(v)), replace=False)
        out["features"][col] = {
            "edges": np.quantile(v, np.linspace(0, 1, 11)).tolist(),
            "mean": float(v.mean()),
            "p50": float(np.median(v)),
            "sample": sample.tolist(),
        }
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(out, indent=2), encoding="utf-8")


def load_profile() -> dict | None:
    return json.loads(PROFILE.read_text(encoding="utf-8")) if PROFILE.exists() else None


def report(rows: list[dict], cases: dict, profile: dict | None) -> dict:
    n = len(rows)
    out = {
        "n": n,
        "flag_rate": float(np.mean([r["decision"] == "flag" for r in rows])) if rows else None,
        "score": None,
        "drift": {},
        "cases": cases,
        "injected": sum(1 for r in rows if r.get("source") == "injected"),
        "live": sum(1 for r in rows if r.get("source") != "injected"),
    }
    if rows:
        scores = np.array([r["score"] for r in rows], dtype=float)
        out["score"] = {
            "mean": float(scores.mean()),
            "p50": float(np.quantile(scores, 0.5)),
            "p90": float(np.quantile(scores, 0.9)),
            "p99": float(np.quantile(scores, 0.99)),
        }
        counts, edges = np.histogram(scores, bins=np.linspace(0, 1, 11))
        out["score_histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
    if profile and n >= 30:
        df = pd.DataFrame([r["features"] for r in rows])
        for col in DRIFT_FEATURES:
            if col not in df:
                continue
            actual = df[col].to_numpy(dtype=float)
            reference = np.array(profile["features"][col]["sample"], dtype=float)
            out["drift"][col] = {"psi": psi(reference, actual), "ks": ks(reference, actual)}
    else:
        out["drift_note"] = "need at least 30 logged requests for drift"
    return out
