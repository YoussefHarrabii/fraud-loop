"""Seed one flagged stream transaction so the demo opens with a case."""

import pandas as pd

from fraud import case_facts, features, serving, store

store.init_db()
bundle = serving.Bundle()
df = features.load()
candidates = df[(df.split == "stream") & (df.is_fraud == 1)].sort_values("amt", ascending=False)

for _, r in candidates.head(200).iterrows():
    history = store.get_history(str(r.cc_num), r.ts)
    row = serving.build_one(history, r)
    out = bundle.score(row)
    if out["decision"] != "flag":
        continue
    rid = store.log_request(r, out, row, bundle.card["created_at"])
    facts = case_facts.build_facts(r, history, out, row)
    cid = store.create_case(rid, r.cc_num, facts, case_facts.template_summary(facts), "template", bundle.card["created_at"])
    print(f"seeded case {cid} from request {rid}, score {out['score']:.4f}, amount {r.amt:.2f}")
    break
else:
    print("no flagged stream fraud in the sample")

for _, r in df[(df.split == "stream") & (df.is_fraud == 0)].head(9).iterrows():
    history = store.get_history(str(r.cc_num), r.ts)
    row = serving.build_one(history, r)
    out = bundle.score(row)
    store.log_request(r, out, row, bundle.card["created_at"])
print("seeded 9 legitimate requests")
