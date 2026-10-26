"""Check that the one-row feature builder matches the batch table on sampled holdout rows."""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud import features, serving

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "processed" / "transactions.parquet"
N = 300

raw = pd.read_parquet(RAW)[["ts", "cc_num", "trans_num", "amt", "category", "merchant", "state", "city_pop"]]
raw["ts"] = pd.to_datetime(raw["ts"])
raw = raw.sort_values(["ts", "trans_num"]).reset_index(drop=True)
by_card = dict(tuple(raw.groupby("cc_num")))

feat = features.load()
sample = feat[feat.split == "holdout"].sample(n=N, random_state=42)
max_diff = {c: 0.0 for c in features.NUMERIC}
cat_bad = {c: 0 for c in features.CATEGORICAL}

for _, r in sample.iterrows():
    card = by_card[r.cc_num]
    hist = card[(card.ts < r.ts) | ((card.ts == r.ts) & (card.trans_num < r.trans_num))]
    row = serving.build_one(hist, r)
    for col in features.NUMERIC:
        a, b = float(r[col]), float(row[col])
        if np.isnan(a) and np.isnan(b):
            continue
        max_diff[col] = max(max_diff[col], abs(a - b))
    for col in features.CATEGORICAL:
        if str(r[col]) != str(row[col]):
            cat_bad[col] += 1

print("max numeric diff:")
for col, v in max_diff.items():
    print(f"  {col:28s} {v:.6g}")
print("categorical mismatches:", cat_bad)
tol = {"secs_since_same_merchant": 5.0, "card_age_days": 1e-4}
for col, v in max_diff.items():
    assert v <= tol.get(col, 1e-4), f"{col} parity failed: {v}"
assert sum(cat_bad.values()) == 0, "categorical parity failed"

print("\nfeature parity OK on", N, "holdout rows")

bundle = serving.Bundle()
for _, r in sample.head(10).iterrows():
    card = by_card[r.cc_num]
    hist = card[(card.ts < r.ts) | ((card.ts == r.ts) & (card.trans_num < r.trans_num))]
    row = serving.build_one(hist, r)
    got = bundle.score(row)["raw_score"]
    want = float(bundle.model.predict(bundle.frame(row))[0])
    assert abs(got - want) < 1e-6, (got, want)
print("bundle scoring OK on 10 rows")
