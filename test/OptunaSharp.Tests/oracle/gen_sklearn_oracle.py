"""Regenerate the committed byte-exact scikit-learn / optuna_integration oracle.

Captures real scikit-learn 1.8.0 + optuna_integration output so the .NET port can be replayed against
it with NO Python at test time (the OptunaSharp.Tests.Oracle model, as the LightGBM tuner does). Run:

    python test/OptunaSharp.Tests.Integration.Sklearn/oracle/gen_sklearn_oracle.py

All floats go through json's repr (shortest round-trip), so System.Text.Json parses the identical
float64 bit patterns; the .NET assertions are exact (==). Values are host-sensitive (float32 forest
accumulation + platform np.int_ bootstrap width), so regenerate on the same x64 host the tests pin to.
"""
import json
import os
import numpy as np
import sklearn
import optuna
import optuna_integration
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from optuna_integration import OptunaSearchCV
from optuna.distributions import IntDistribution


def make_xy(seed, n, d):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # a smooth, learnable target with a nonlinear term
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * (X[:, 2] * X[:, 2])
    if d > 3:
        y = y + 0.3 * X[:, 3]
    y = y + rng.standard_normal(n) * 0.1
    return X, y


def jf(x):
    return float(x)


def component(name, seed, n, d, scoring="r2", **rf):
    X, y = make_xy(seed, n, d)

    def mk():
        return RandomForestRegressor(**rf)

    preds = mk().fit(X, y).predict(X)
    r2 = mk().fit(X, y).score(X, y)
    folds = [te.tolist() for _, te in KFold(5).split(X)]
    cv = cross_validate(mk(), X, y, cv=KFold(5), scoring=scoring, return_train_score=True)
    return {
        "name": name,
        "X": [[jf(v) for v in row] for row in X],
        "y": [jf(v) for v in y],
        "config": rf,
        "scoring": scoring,
        "preds": [jf(p) for p in preds],
        "r2": jf(r2),
        "folds": folds,
        "cv_test": [jf(v) for v in cv["test_score"]],
        "cv_train": [jf(v) for v in cv["train_score"]],
        "cv_mean": jf(np.nanmean(cv["test_score"])),
    }


def search(name, seed, n, d, dists, random_state, n_trials, cv):
    X, y = make_xy(seed, n, d)
    dist_objs = {k: IntDistribution(lo, hi) for k, (lo, hi) in dists.items()}
    s = OptunaSearchCV(
        RandomForestRegressor(random_state=0),
        dist_objs, cv=cv, n_trials=n_trials, random_state=random_state, refit=True,
        return_train_score=True,
    )
    s.fit(X, y)
    trials = []
    for t in s.trials_:
        ua = t.user_attrs
        splits = [jf(ua[f"split{i}_test_score"]) for i in range(s.n_splits_)]
        trials.append({
            "params": {k: int(v) for k, v in t.params.items()},
            "mean_test_score": jf(ua["mean_test_score"]),
            "std_test_score": jf(ua["std_test_score"]),
            "splits": splits,
        })
    return {
        "name": name,
        "X": [[jf(v) for v in row] for row in X],
        "y": [jf(v) for v in y],
        "dists": [{"name": k, "low": lo, "high": hi} for k, (lo, hi) in dists.items()],
        "random_state": random_state,
        "n_trials": n_trials,
        "cv": cv,
        "n_splits": s.n_splits_,
        "best_score": jf(s.best_score_),
        "best_params": {k: int(v) for k, v in s.best_params_.items()},
        "trials": trials,
    }


oracle = {
    "sklearn_version": sklearn.__version__,
    "optuna_integration_version": optuna_integration.__version__,
    "component": [
        component("base", 123, 40, 4, n_estimators=7, max_depth=4, random_state=0),
        component("sqrt_features", 5, 50, 5, n_estimators=10, max_depth=5, max_features="sqrt", random_state=1),
        component("frac_features", 9, 45, 6, n_estimators=8, max_depth=6, max_features=0.5, random_state=2),
        component("no_bootstrap", 3, 36, 4, n_estimators=6, max_depth=3, bootstrap=False, random_state=3),
        component("min_samples", 11, 55, 4, n_estimators=9, min_samples_split=5, min_samples_leaf=3, random_state=4),
        component("unlimited_depth", 7, 30, 3, n_estimators=5, random_state=5),
        component("neg_mse", 21, 40, 4, scoring="neg_mean_squared_error", n_estimators=7, max_depth=4, random_state=0),
    ],
    "search": [
        search("depth_estimators", 5, 50, 3,
               {"n_estimators": (5, 20), "max_depth": (2, 8)}, random_state=42, n_trials=8, cv=5),
        search("with_min_split", 17, 60, 4,
               {"n_estimators": (5, 15), "max_depth": (2, 10), "min_samples_split": (2, 12)},
               random_state=7, n_trials=10, cv=3),
    ],
}

out = os.path.join(os.path.dirname(__file__), "sklearn_oracle.json")
with open(out, "w") as f:
    json.dump(oracle, f, indent=1)
print("wrote", out)
print("sklearn", sklearn.__version__, "optuna_integration", optuna_integration.__version__)
print("component scenarios:", len(oracle["component"]), "search scenarios:", len(oracle["search"]))
for s in oracle["search"]:
    print("  search", s["name"], "best", repr(s["best_score"]), s["best_params"])
