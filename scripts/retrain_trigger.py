"""P4 retrain trigger and promotion gate.

Trigger: live drift (PSI > 0.25), a live precision drop, or --force.
Gate: candidate must be cheaper on the fixed holdout with a paired bootstrap CI
below zero, and must not lose more than 2 recall points.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from fraud import features, monitoring, policy, serving, store
from fraud.train import FALSE_ALARM_COST, RECOVERY, fit_lgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DOCS" / "p4-retrain.md"


def metrics(y, p, amt, threshold):
    flag = p >= threshold
    tp, fp, fn = flag & (y == 1), flag & (y == 0), (~flag) & (y == 1)
    cost = amt[fn].sum() * RECOVERY + fp.sum() * FALSE_ALARM_COST
    return {
        "recall": float(tp.sum() / (y == 1).sum()),
        "precision": float(y[flag].mean()) if flag.any() else 0.0,
        "flags_per_day": float(flag.sum() / 30),
        "fp_per_day": float(fp.sum() / 30),
        "amount_recall": float(amt[tp].sum() / amt[y == 1].sum()),
        "cost_1k": float(cost / len(y) * 1000),
    }


def trigger_reasons(force: bool) -> list[str]:
    if force:
        return ["forced"]
    live = monitoring.report(store.request_features(500), store.case_stats(), monitoring.load_profile())
    reasons = [f"PSI {name} {v['psi']:.2f}" for name, v in live["drift"].items() if v["psi"] and v["psi"] > 0.25]
    cases = live["cases"]
    if cases["reviewed"] >= 20 and cases["precision"] is not None and cases["precision"] < 0.5:
        reasons.append(f"live precision {cases['precision']:.2f}")
    if live["n"] >= 100 and live["flag_rate"] is not None and not 0.003 <= live["flag_rate"] <= 0.03:
        reasons.append(f"flag rate {live['flag_rate']:.2%}")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidate", default="configs/candidate_model.json")
    args = parser.parse_args()

    reasons = trigger_reasons(args.force)
    print("trigger reasons:", reasons or "none")
    if not reasons:
        print("no retrain")
        return

    cand_cfg = json.loads((ROOT / args.candidate).read_text(encoding="utf-8"))
    df = features.load()
    tr, va, ho = df[df.split == "train"], df[df.split == "val"], df[df.split == "holdout"]
    Xtr, Xva, Xho = (d[features.NUMERIC + features.CATEGORICAL] for d in (tr, va, ho))
    ytr, yva, yho = (d.is_fraud.to_numpy() for d in (tr, va, ho))
    aho = ho.amt.to_numpy()

    cand = fit_lgb(cand_cfg["params"], Xtr, ytr, cand_cfg["rounds"])
    cva_raw = cand.predict(Xva, num_iteration=cand_cfg["rounds"])
    ccal = policy.fit_platt(cva_raw, yva)
    cva = policy.apply_platt(ccal, cva_raw)
    cho = policy.apply_platt(ccal, cand.predict(Xho, num_iteration=cand_cfg["rounds"]))
    cth = policy.recall_threshold(cva, yva, cand_cfg["target_recall"])
    cm = metrics(yho, cho, aho, cth)

    prod = serving.Bundle()
    pho = policy.apply_platt(prod.cal, prod.model.predict(Xho))
    pm = metrics(yho, pho, aho, prod.threshold)

    ci = policy.bootstrap_cost_diff(
        yho, cho, pho, aho, cth, prod.threshold, FALSE_ALARM_COST, RECOVERY
    )
    promote = policy.should_promote(ci["high"], cm["recall"], pm["recall"])

    table = pd.DataFrame({"candidate": cm, "production": pm}).T
    print(f"candidate threshold {cth:.6f}; production threshold {prod.threshold:.6f}")
    print(table.round(4).to_string())
    print(f"cost diff mean {ci['mean']:.2f}, 95% CI [{ci['low']:.2f}, {ci['high']:.2f}], promote {promote}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cdir = ROOT / "artifacts" / "candidates" / stamp
    cdir.mkdir(parents=True, exist_ok=True)
    cand.save_model(str(cdir / "model.txt"))
    joblib.dump(ccal, cdir / "calibrator.joblib")
    card = {
        "created_at": stamp, "params": cand_cfg["params"], "rounds": cand_cfg["rounds"],
        "target_recall": cand_cfg["target_recall"], "threshold": cth,
        "holdout": cm, "production": pm,
        "promotion": {"promote": promote, "cost_diff_mean": ci["mean"],
                      "ci_low": ci["low"], "ci_high": ci["high"], "reasons": reasons},
    }
    (cdir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("fraud-p4")
    with mlflow.start_run(run_name="retrain-gate"):
        mlflow.log_params({**cand_cfg["params"], "rounds": cand_cfg["rounds"],
                           "threshold": cth, "target_recall": cand_cfg["target_recall"]})
        mlflow.log_metrics({f"cand_{k}": v for k, v in cm.items()})
        mlflow.log_metrics({f"prod_{k}": v for k, v in pm.items()})
        mlflow.log_metrics({"cost_diff_mean": ci["mean"], "ci_low": ci["low"],
                            "ci_high": ci["high"], "promote": int(promote)})
        mlflow.lightgbm.log_model(cand, name="candidate")

    OUT.write_text(
        "\n".join([
            "# P4 retrain and promotion gate", "",
            f"Trigger: {', '.join(reasons)}", "",
            "```", table.round(4).to_string(), "```", "",
            f"Candidate threshold {cth:.6f}, production threshold {prod.threshold:.6f}.",
            f"Paired bootstrap cost difference mean {ci['mean']:.2f}, 95% CI [{ci['low']:.2f}, {ci['high']:.2f}].",
            f"Promote: {promote}. Rule: CI upper bound below zero and recall within 2 points of production.",
            "", f"Candidate artifacts: `artifacts/candidates/{stamp}`. Production is unchanged.",
        ]), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
