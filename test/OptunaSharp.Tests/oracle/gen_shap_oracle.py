#!/usr/bin/env python3
"""Regenerate shap_oracle.json — the committed byte-exact oracle for ShapleyImportanceEvaluator.

Runs real Optuna + the actual optuna_integration.shap.ShapleyImportanceEvaluator pipeline
(shap.TreeExplainer over sklearn.ensemble.RandomForestRegressor) and records, per scenario, the
resulting importances as raw IEEE-754 bits so OptunaSharp.Tests.Integration.Shap can assert
bit-for-bit equality with NO Python at test time.

Pinned reference: optuna 5.0.0.dev (vendored, `pip install -e src/optuna`), optuna_integration
5.1.0.dev, shap 0.52.0, scikit-learn 1.8.0, numpy 2.4.2 (x64). Rerun after bumping any of those:

    python test/OptunaSharp.Tests.Integration.Shap/oracle/gen_shap_oracle.py
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna_integration.shap import ShapleyImportanceEvaluator


def objective(trial):
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_int("x3", 2, 4, log=True)
    x4 = trial.suggest_categorical("x4", [0.1, 1.0, 10.0])
    return x1 + x2 * x3 + x4


# name, sampler_seed, n_trials, eval_seed, n_trees, max_depth
SCENARIOS = [
    dict(name="default",       sampler_seed=0, n_trials=8,   eval_seed=0, n_trees=64, max_depth=64),
    dict(name="ntrees10",      sampler_seed=0, n_trials=8,   eval_seed=0, n_trees=10, max_depth=64),
    dict(name="ntrees20_d2",   sampler_seed=0, n_trials=8,   eval_seed=0, n_trees=20, max_depth=2),
    dict(name="maxdepth1",     sampler_seed=0, n_trials=8,   eval_seed=0, n_trees=64, max_depth=1),
    dict(name="mid40",         sampler_seed=3, n_trials=40,  eval_seed=3, n_trees=64, max_depth=64),
    dict(name="large150",      sampler_seed=7, n_trials=150, eval_seed=7, n_trees=64, max_depth=64),
    dict(name="large200_nt37", sampler_seed=1, n_trials=200, eval_seed=5, n_trees=37, max_depth=9),
]


def main() -> None:
    out = []
    for sc in SCENARIOS:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=sc["sampler_seed"]))
        study.optimize(objective, n_trials=sc["n_trials"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = ShapleyImportanceEvaluator(
                n_trees=sc["n_trees"], max_depth=sc["max_depth"], seed=sc["eval_seed"]
            )
        res = ev.evaluate(study)
        sc = dict(sc)
        sc["expected"] = [
            dict(param=k, bits=int(np.float64(v).view(np.int64)), value=float(v))
            for k, v in res.items()
        ]
        out.append(sc)

    path = os.path.join(os.path.dirname(__file__), "shap_oracle.json")
    with open(path, "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {len(out)} scenarios -> {path}")


if __name__ == "__main__":
    main()
