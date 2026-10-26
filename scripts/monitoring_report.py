"""Offline drift and holdout operating report. Writes DOCS/p4-monitoring.md."""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud import features, monitoring, policy, serving, store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DOCS" / "p4-monitoring.md"

df = features.load()
ref = df[df.split == "train"]
cur = df[df.split == "holdout"]

rows = []
for col in monitoring.DRIFT_FEATURES:
    p = monitoring.psi(ref[col].dropna().to_numpy(float), cur[col].dropna().to_numpy(float))
    k = monitoring.ks(ref[col].dropna().to_numpy(float), cur[col].dropna().to_numpy(float))
    verdict = "stable" if p is not None and p < monitoring.STABLE else "moderate" if p is not None and p < monitoring.MODERATE else "significant"
    rows.append({"feature": col, "psi": p, "ks": k, "verdict": verdict})

bundle = serving.Bundle()
X = cur[features.NUMERIC + features.CATEGORICAL]
p = policy.apply_platt(bundle.cal, bundle.model.predict(X))
flag = p >= bundle.threshold
scores = pd.Series(p)

cases = store.case_stats()
table = pd.DataFrame(rows).set_index("feature")
print(table.round(4).to_string())
print(f"holdout flag rate {flag.mean():.4%}, score p50 {scores.quantile(.5):.4f}, p99 {scores.quantile(.99):.4f}")
print("cases", cases)

lines = [
    "# P4 monitoring",
    "",
    "Reference: train (Jan 2019 to Sep 2020). Current: holdout (Nov 2020).",
    "",
    "| feature | PSI | KS | verdict |",
    "|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['feature']} | {r['psi']:.4f} | {r['ks']:.4f} | {r['verdict']} |")
lines += [
    "",
    f"Holdout flag rate: {flag.mean():.4%}. Score p50 {scores.quantile(.5):.4f}, p99 {scores.quantile(.99):.4f}.",
    "",
    "PSI: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant.",
    "",
    f"Analyst decisions so far: {cases}. Precision on reviewed cases is descriptive, not a full live precision, because only flagged cases are reviewed.",
    "",
    "Live endpoint: `GET /monitoring`. It uses the last 500 logged requests against the saved training profile and reports `drift_note` until 30 requests exist.",
]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
