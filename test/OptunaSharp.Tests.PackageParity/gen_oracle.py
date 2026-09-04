#!/usr/bin/env python3
"""
Generate parity oracles from REAL Optuna driving REAL ML packages.

Each oracle records the exact trial trajectory (search space + per-trial params + the value the ML
package produced) of a seeded study. The PACKAGED OptunaSharp then replays that trajectory (see
OracleReplay.cs) and must reproduce the same suggestions — bit-exact for RandomSampler, to tolerance
for TPE.

Output dir: $PACKAGE_PARITY_ORACLE_DIR (default ./oracle). Deps: optuna xgboost lightgbm scikit-learn.
"""
import json
import os

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

OUT = os.environ.get("PACKAGE_PARITY_ORACLE_DIR", "oracle")
os.makedirs(OUT, exist_ok=True)


def suggest_all(trial, space):
    """Make the suggest_* calls in space order — the identical order OptunaSharp replays."""
    values = {}
    for e in space:
        if e["kind"] == "float":
            values[e["name"]] = trial.suggest_float(e["name"], e["low"], e["high"], log=e.get("log", False))
        else:
            values[e["name"]] = trial.suggest_int(e["name"], int(e["low"]), int(e["high"]))
    return values


def make_sampler(kind, seed):
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def emit(name, kind, seed, direction, space, objective, n_trials):
    study = optuna.create_study(direction=direction, sampler=make_sampler(kind, seed))
    study.optimize(objective, n_trials=n_trials)
    trials = [
        {
            "number": t.number,
            "state": int(t.state),
            "params": dict(t.params),
            "value": None if t.value is None else float(t.value),
        }
        for t in study.trials
    ]
    record = {
        "name": name, "sampler": kind, "seed": seed, "direction": direction,
        "space": space, "trials": trials,
    }
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(record, f)
    print(f"[oracle] {name}: {len(trials)} trials, best={study.best_value:.6g}")


# --------------------------------------------------------------------- shared regression data
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------- xgboost (real) + RandomSampler
import xgboost as xgb

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dtest = xgb.DMatrix(X_te, label=y_te)

# The EXACT 8-parameter grid of bexgboost/823df08d2e3760538e9b931d38439a68 (16 stars) — the same
# search space examples/XGBoostHpo.cs mirrors. Param names match XGBoost's own, so they spread
# straight into the training params dict (as the gist does). The gist's XGBoostPruningCallback is
# omitted: RandomSampler suggestions are value-independent, so pruning wouldn't change this parity.
XGB_SPACE = [
    {"kind": "float", "name": "eta", "low": 0.01, "high": 0.3, "log": False},
    {"kind": "int", "name": "max_depth", "low": 3, "high": 10},
    {"kind": "float", "name": "subsample", "low": 0.5, "high": 1.0, "log": False},
    {"kind": "float", "name": "colsample_bytree", "low": 0.5, "high": 1.0, "log": False},
    {"kind": "float", "name": "gamma", "low": 0.0, "high": 10.0, "log": False},
    {"kind": "float", "name": "min_child_weight", "low": 0.1, "high": 10.0, "log": False},
    {"kind": "float", "name": "lambda", "low": 0.1, "high": 10.0, "log": False},
    {"kind": "float", "name": "alpha", "low": 0.0, "high": 10.0, "log": False},
]


def xgb_objective(trial):
    v = suggest_all(trial, XGB_SPACE)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
        "nthread": 1, "seed": 0, **v,
    }
    model = xgb.train(params, dtrain, num_boost_round=50)
    return rmse(y_te, model.predict(dtest))


emit("xgboost-random-seed0", "random", 0, "minimize", XGB_SPACE, xgb_objective, 25)
emit("xgboost-random-seed42", "random", 42, "minimize", XGB_SPACE, xgb_objective, 25)


# --------------------------------------------------------------------- lightgbm (real) + RandomSampler
import lightgbm as lgb

ltrain = lgb.Dataset(X_tr, label=y_tr)

LGB_SPACE = [
    {"kind": "float", "name": "learning_rate", "low": 0.01, "high": 0.3, "log": True},
    {"kind": "int", "name": "num_leaves", "low": 8, "high": 128},
    {"kind": "int", "name": "max_depth", "low": 3, "high": 12},
    {"kind": "float", "name": "feature_fraction", "low": 0.5, "high": 1.0, "log": False},
    {"kind": "float", "name": "bagging_fraction", "low": 0.5, "high": 1.0, "log": False},
]


def lgb_objective(trial):
    v = suggest_all(trial, LGB_SPACE)
    params = {
        "objective": "regression", "metric": "rmse", "verbosity": -1, "num_threads": 1, "seed": 0,
        "learning_rate": v["learning_rate"], "num_leaves": v["num_leaves"], "max_depth": v["max_depth"],
        "feature_fraction": v["feature_fraction"], "bagging_fraction": v["bagging_fraction"], "bagging_freq": 1,
    }
    model = lgb.train(params, ltrain, num_boost_round=40)
    return rmse(y_te, model.predict(X_te))


emit("lightgbm-random-seed0", "random", 0, "minimize", LGB_SPACE, lgb_objective, 25)


# --------------------------------------------------------------------- sklearn SVM (float-only) + TPE
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

Xi, yi = load_iris(return_X_y=True)

SVM_SPACE = [
    {"kind": "float", "name": "C", "low": 1e-3, "high": 1e3, "log": True},
    {"kind": "float", "name": "gamma", "low": 1e-4, "high": 1e1, "log": True},
]


def svm_objective(trial):
    v = suggest_all(trial, SVM_SPACE)
    clf = SVC(C=v["C"], gamma=v["gamma"])
    return float(cross_val_score(clf, Xi, yi, cv=3).mean())


emit("svm-tpe-seed0", "tpe", 0, "maximize", SVM_SPACE, svm_objective, 30)
emit("svm-tpe-seed42", "tpe", 42, "maximize", SVM_SPACE, svm_objective, 30)

print("all oracles written to", OUT)
