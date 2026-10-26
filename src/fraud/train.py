"""P1: logistic baseline, LightGBM vs XGBoost with Optuna, cost threshold, MLflow."""

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud import features

ROOT = Path(__file__).resolve().parents[2]
FALSE_ALARM_COST = 7.0
RECOVERY = 0.8
THREADS = 8
SEED = 42


def evaluate(y, p, amt) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    amt = np.asarray(amt)
    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, 400)))
    thresholds = thresholds[thresholds > 0]
    if len(thresholds) == 0:
        thresholds = np.array([1.0])
    costs = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        flag = p >= t
        costs[i] = (amt[(~flag) & (y == 1)] * RECOVERY).sum() + ((flag) & (y == 0)).sum() * FALSE_ALARM_COST
    best = int(np.argmin(costs))
    th = float(thresholds[best])
    flag = p >= th
    row_cost = np.where(flag, np.where(y == 1, 0.0, FALSE_ALARM_COST), np.where(y == 1, amt * RECOVERY, 0.0))
    rng = np.random.default_rng(SEED)
    boots = np.empty(300)
    for i in range(300):
        idx = rng.integers(0, len(y), len(y))
        boots[i] = row_cost[idx].sum()
    rule_flag = p * amt * RECOVERY > FALSE_ALARM_COST
    rule_cost = (amt[(~rule_flag) & (y == 1)] * RECOVERY).sum() + (rule_flag & (y == 0)).sum() * FALSE_ALARM_COST
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "threshold": th,
        "cost": float(costs[best]),
        "cost_per_1k": float(costs[best] / len(y) * 1000),
        "cost_ci_low": float(np.percentile(boots, 2.5)),
        "cost_ci_high": float(np.percentile(boots, 97.5)),
        "flag_rate": float(flag.mean()),
        "precision": float(y[flag].mean()) if flag.any() else 0.0,
        "recall": float(flag[y == 1].mean()) if (y == 1).any() else 0.0,
        "cost_no_flags": float((amt[y == 1] * RECOVERY).sum()),
        "cost_all_flags": float((y == 0).sum() * FALSE_ALARM_COST),
        "rule_flag_rate": float(rule_flag.mean()),
        "rule_cost_per_1k": float(rule_cost / len(y) * 1000),
        "rule_precision": float(y[rule_flag].mean()) if rule_flag.any() else 0.0,
        "rule_recall": float(rule_flag[y == 1].mean()) if (y == 1).any() else 0.0,
    }


def tune_lgb(Xtr, ytr, Xva, yva, n_trials: int, run_name: str) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "first_metric_only": True,
            "verbosity": -1,
            "num_threads": THREADS,
            "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        }
        dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=features.CATEGORICAL)
        dval = lgb.Dataset(Xva, label=yva, reference=dtrain)
        t0 = time.perf_counter()
        model = lgb.train(
            params, dtrain, num_boost_round=1500, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        p = model.predict(Xva, num_iteration=model.best_iteration)
        score = average_precision_score(yva, p)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_pr_auc", score)
            mlflow.log_metric("best_iteration", model.best_iteration)
            mlflow.log_metric("seconds", time.perf_counter() - t0)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


def tune_xgb(Xtr, ytr, Xva, yva, n_trials: int, run_name: str) -> optuna.Study:
    dval = xgb.DMatrix(Xva, label=yva, enable_categorical=True, nthread=THREADS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_cat_to_onehot": 4,
            "nthread": THREADS,
            "seed": SEED,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        }
        dtrain = xgb.DMatrix(Xtr, label=ytr, enable_categorical=True, nthread=THREADS)
        t0 = time.perf_counter()
        model = xgb.train(
            params, dtrain, num_boost_round=1500, evals=[(dval, "val")], verbose_eval=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=50, metric_name="aucpr", maximize=True, save_best=True)],
        )
        p = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        score = average_precision_score(yva, p)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_pr_auc", score)
            mlflow.log_metric("best_iteration", model.best_iteration)
            mlflow.log_metric("seconds", time.perf_counter() - t0)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


FIXED_LGB = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "num_threads": THREADS,
    "seed": SEED,
}
FIXED_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_cat_to_onehot": 4,
    "nthread": THREADS,
    "seed": SEED,
}


def fit_lgb(params, X, y, rounds):
    dtrain = lgb.Dataset(X, label=y, categorical_feature=features.CATEGORICAL)
    return lgb.train({**FIXED_LGB, **params}, dtrain, num_boost_round=rounds)


def fit_xgb(params, X, y, rounds):
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True, nthread=THREADS)
    return xgb.train({**FIXED_XGB, **params}, dtrain, num_boost_round=rounds)


def make_logistic():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cats = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)),
    ])
    prep = ColumnTransformer([("num", numeric, features.NUMERIC), ("cat", cats, features.BASELINE_CAT)])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, solver="lbfgs"))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("fraud-p1")

    df = features.load(rebuild=args.rebuild)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    if args.smoke:
        train = train.sample(n=min(200_000, len(train)), random_state=SEED)
        val = val.sample(n=min(50_000, len(val)), random_state=SEED)
    print(f"train {len(train):,} ({int(train.is_fraud.sum()):,} fraud) | val {len(val):,} ({int(val.is_fraud.sum()):,} fraud)")

    Xtr = train[features.NUMERIC + features.CATEGORICAL].copy()
    Xva = val[features.NUMERIC + features.CATEGORICAL].copy()
    ytr, yva = train.is_fraud.to_numpy(), val.is_fraud.to_numpy()
    amt_va = val.amt.to_numpy()

    results = {}
    fitted = {}
    with mlflow.start_run(run_name="logistic"):
        t0 = time.perf_counter()
        model = make_logistic()
        cat_tr = train[features.BASELINE_CAT].astype(str)
        cat_va = val[features.BASELINE_CAT].astype(str)
        model.fit(pd.concat([train[features.NUMERIC], cat_tr], axis=1), ytr)
        p = model.predict_proba(pd.concat([val[features.NUMERIC], cat_va], axis=1))[:, 1]
        m = evaluate(yva, p, amt_va)
        m["seconds"] = time.perf_counter() - t0
        results["logistic"] = m
        fitted["logistic"] = (model, None)
        mlflow.log_params({"model": "logistic", "max_iter": 1000, "smoke": args.smoke})
        mlflow.log_metrics(m)

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params({"model": "lightgbm", "trials": args.trials, "smoke": args.smoke})
        study = tune_lgb(Xtr, ytr, Xva, yva, args.trials, "lightgbm")
        best = study.best_params
        rounds = int(study.best_trial.user_attrs["best_iteration"])
        model = fit_lgb(best, Xtr, ytr, rounds)
        p = model.predict(Xva, num_iteration=rounds)
        m = evaluate(yva, p, amt_va)
        m["seconds"] = sum(t.user_attrs.get("seconds", 0) for t in study.trials)
        results["lightgbm"] = m
        fitted["lightgbm"] = (model, rounds)
        with mlflow.start_run(run_name="best", nested=True):
            mlflow.log_params({**best, "rounds": rounds})
            mlflow.log_metrics(m)
            mlflow.lightgbm.log_model(model, name="model")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({"model": "xgboost", "trials": args.trials, "smoke": args.smoke})
        study = tune_xgb(Xtr, ytr, Xva, yva, args.trials, "xgboost")
        best = study.best_params
        rounds = int(study.best_trial.user_attrs["best_iteration"]) + 1
        model = fit_xgb(best, Xtr, ytr, rounds)
        dval = xgb.DMatrix(Xva, enable_categorical=True, nthread=THREADS)
        p = model.predict(dval, iteration_range=(0, rounds))
        m = evaluate(yva, p, amt_va)
        m["seconds"] = sum(t.user_attrs.get("seconds", 0) for t in study.trials)
        results["xgboost"] = m
        fitted["xgboost"] = (model, rounds)
        with mlflow.start_run(run_name="best", nested=True):
            mlflow.log_params({**best, "rounds": rounds})
            mlflow.log_metrics(m)
            mlflow.xgboost.log_model(model, name="model")

    cols = ["pr_auc", "roc_auc", "cost_per_1k", "rule_cost_per_1k", "rule_flag_rate",
            "rule_precision", "rule_recall", "flag_rate", "precision", "recall",
            "cost_ci_low", "cost_ci_high", "seconds"]
    table = pd.DataFrame(results).T[cols]
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(f"\nreferences: no flags {results['lightgbm']['cost_no_flags']:,.0f}, all flags {results['lightgbm']['cost_all_flags']:,.0f}")
    print("global thresholds:", {k: round(v["threshold"], 4) for k, v in results.items()})

    winner = min(results, key=lambda k: results[k]["rule_cost_per_1k"])
    print(f"\nwinner on validation rule cost: {winner}")

    holdout = df[df.split == "holdout"]
    Xho = holdout[features.NUMERIC + features.CATEGORICAL]
    yd = holdout.is_fraud.to_numpy()
    w_model, w_rounds = fitted[winner]
    if winner == "logistic":
        p = w_model.predict_proba(pd.concat([holdout[features.NUMERIC], holdout[features.BASELINE_CAT].astype(str)], axis=1))[:, 1]
    elif winner == "lightgbm":
        p = w_model.predict(Xho, num_iteration=w_rounds)
    else:
        p = w_model.predict(xgb.DMatrix(Xho, enable_categorical=True, nthread=THREADS), iteration_range=(0, w_rounds))
    amt_ho = holdout.amt.to_numpy()
    rule_flag = p * amt_ho * RECOVERY > FALSE_ALARM_COST
    ho_cost = (amt_ho[(~rule_flag) & (yd == 1)] * RECOVERY).sum() + (rule_flag & (yd == 0)).sum() * FALSE_ALARM_COST
    th = results[winner]["threshold"]
    flag = p >= th
    th_cost = (amt_ho[(~flag) & (yd == 1)] * RECOVERY).sum() + (flag & (yd == 0)).sum() * FALSE_ALARM_COST
    print(f"holdout rule: PR-AUC {average_precision_score(yd, p):.4f}, cost/1k {ho_cost / len(yd) * 1000:,.2f}, "
          f"flag rate {rule_flag.mean():.4f}, precision {yd[rule_flag].mean():.3f}, recall {rule_flag[yd == 1].mean():.3f}")
    print(f"holdout global threshold {th:.4f}: cost/1k {th_cost / len(yd) * 1000:,.2f}, flag rate {flag.mean():.4f}")
    with mlflow.start_run(run_name="holdout"):
        mlflow.log_params({"winner": winner, "global_threshold": th})
        mlflow.log_metrics({
            "holdout_pr_auc": average_precision_score(yd, p),
            "holdout_rule_cost_per_1k": ho_cost / len(yd) * 1000,
            "holdout_rule_flag_rate": float(rule_flag.mean()),
            "holdout_rule_precision": float(yd[rule_flag].mean()),
            "holdout_rule_recall": float(rule_flag[yd == 1].mean()),
            "holdout_threshold_cost_per_1k": th_cost / len(yd) * 1000,
            "holdout_threshold_flag_rate": float(flag.mean()),
        })


if __name__ == "__main__":
    main()
