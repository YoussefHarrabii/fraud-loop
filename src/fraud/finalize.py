"""Train the chosen model, calibrate it, pick the recall threshold, save the serving bundle."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from fraud import features, policy
from fraud.train import fit_lgb

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "final_model.json"
ART = ROOT / "artifacts" / "final"
OUT = ROOT / "DOCS" / "final-model.md"


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    df = features.load()
    tr, va, ho = df[df.split == "train"], df[df.split == "val"], df[df.split == "holdout"]
    Xtr, Xva, Xho = (d[features.NUMERIC + features.CATEGORICAL] for d in (tr, va, ho))
    ytr, yva, yho = (d.is_fraud.to_numpy() for d in (tr, va, ho))
    ava, aho = va.amt.to_numpy(), ho.amt.to_numpy()

    model = fit_lgb(cfg["params"], Xtr, ytr, cfg["rounds"])
    raw_va = model.predict(Xva, num_iteration=cfg["rounds"])
    raw_ho = model.predict(Xho, num_iteration=cfg["rounds"])
    cal = policy.fit_platt(raw_va, yva)
    pva = policy.apply_platt(cal, raw_va)
    pho = policy.apply_platt(cal, raw_ho)
    threshold = policy.recall_threshold(pva, yva, cfg["target_recall"])

    def evaluate(y, p, amt, days):
        flag = policy.decide(p, threshold)
        tp, fp, fn = flag & (y == 1), flag & (y == 0), (~flag) & (y == 1)
        cost = amt[fn].sum() * cfg["recovery"] + fp.sum() * cfg["false_alarm_cost"]
        return {
            "rows": int(len(y)),
            "frauds": int(y.sum()),
            "flags": int(flag.sum()),
            "tp": int(tp.sum()),
            "fp": int(fp.sum()),
            "fn": int(fn.sum()),
            "precision": float(y[flag].mean()) if flag.any() else 0.0,
            "recall": float(tp.sum() / (y == 1).sum()),
            "flag_rate": float(flag.mean()),
            "flags_per_day": float(flag.sum() / days),
            "false_alarms_per_day": float(fp.sum() / days),
            "amount_recall": float(amt[tp].sum() / amt[y == 1].sum()),
            "cost": float(cost),
            "cost_per_1k": float(cost / len(y) * 1000),
        }

    val = evaluate(yva, pva, ava, 31)
    ho = evaluate(yho, pho, aho, 30)

    ART.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ART / "model.txt"))
    joblib.dump(cal, ART / "calibrator.joblib")
    card = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"],
        "params": cfg["params"],
        "rounds": cfg["rounds"],
        "calibration": cfg["calibration"],
        "calibration_month": cfg["calibration_month"],
        "target_recall": cfg["target_recall"],
        "threshold": threshold,
        "decision": "flag if calibrated_score >= threshold",
        "features": {"numeric": features.NUMERIC, "categorical": features.CATEGORICAL},
        "validation": val,
        "holdout": ho,
    }
    (ART / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("fraud-p1")
    with mlflow.start_run(run_name="final-model"):
        mlflow.log_params({"model": cfg["model"], "rounds": cfg["rounds"], "threshold": threshold,
                           "target_recall": cfg["target_recall"], "calibration": cfg["calibration"]})
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        mlflow.log_metrics({f"holdout_{k}": v for k, v in ho.items()})
        mlflow.lightgbm.log_model(model, name="model")
        mlflow.log_artifact(str(ART / "calibrator.joblib"), artifact_path="calibrator")
        mlflow.log_artifact(str(ART / "model_card.json"), artifact_path="card")

    print(f"threshold {threshold:.6f} (target {cfg['target_recall']:.0%} recall on validation)")
    print(pd.DataFrame({"validation": val, "holdout": ho}).to_string())
    print(f"wrote {ART}")

    lines = [
        "# Final model",
        "",
        f"LightGBM, {cfg['rounds']} rounds, Platt calibration, threshold {threshold:.6f}.",
        f"Threshold target: {cfg['target_recall']:.0%} recall on {cfg['calibration_month']}.",
        "Decision: flag if calibrated score is at or above the threshold. Amount is not part of the decision.",
        "",
        "| metric | validation | holdout |",
        "|---|---|---|",
    ]
    for key in ["flags", "tp", "fp", "fn", "precision", "recall", "flag_rate", "flags_per_day",
                "false_alarms_per_day", "amount_recall", "cost_per_1k"]:
        lines.append(f"| {key} | {val[key]:.4f} | {ho[key]:.4f} |" if isinstance(val[key], float) else f"| {key} | {val[key]} | {ho[key]} |")
    lines += ["", "Artifacts: `artifacts/final/model.txt`, `calibrator.joblib`, `model_card.json`."]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
