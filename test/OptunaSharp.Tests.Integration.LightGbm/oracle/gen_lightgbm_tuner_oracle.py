# Regenerates lightgbm_tuner_synth.json — the bit-exact oracle for the LightGBM tuner's stepwise
# search. It monkeypatches lightgbm.train / lightgbm.cv inside optuna_integration with a DETERMINISTIC
# synthetic score of the accumulated LightGBM parameters, so the captured trial sequence depends only on
# Optuna's GridSampler / TPESampler (which OptunaSharp ports bit-for-bit) — not on real LightGBM. The
# synthetic score uses only +,-,* so it is bit-identical between Python and C#.
#
# Prereqs: pip install optuna-integration --no-deps  (optuna 5.x + lightgbm + tqdm already present).
# Run:     python gen_lightgbm_tuner_oracle.py   (writes lightgbm_tuner_synth.json next to this file)
#
# The C# LightGbmTunerParityTests reproduces this exact sequence with a synthetic ILightGbmTrainer.
import json
import os
import types
import warnings

warnings.filterwarnings("ignore")
import optuna

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import optuna_integration.lightgbm as lgb
from optuna_integration.lightgbm._lightgbm_tuner import optimize as _opt


def synth(params):
    ff = params.get("feature_fraction", 1.0)
    nl = float(params.get("num_leaves", 31))
    bf = params.get("bagging_fraction", 1.0)
    bfreq = float(params.get("bagging_freq", 0))
    l1 = params.get("lambda_l1", 0.0)
    l2 = params.get("lambda_l2", 0.0)
    mcs = float(params.get("min_child_samples", 20))
    return (1.0 * (ff - 0.75) * (ff - 0.75)
            + 0.0001 * (nl - 100.0) * (nl - 100.0)
            + 1.0 * (bf - 0.8) * (bf - 0.8)
            + 0.01 * (bfreq - 3.0) * (bfreq - 3.0)
            + 0.1 * l1 + 0.1 * l2
            + 0.0001 * (mcs - 15.0) * (mcs - 15.0))


def resolved_metric(params):
    metric = params.get("metric", "binary_logloss")
    return metric if isinstance(metric, str) else list(metric)[-1]


def fake_train(params, train_set, **kwargs):
    b = types.SimpleNamespace()
    b.best_score = {"valid_0": {resolved_metric(params): synth(params)}}
    b.current_iteration = lambda: kwargs.get("num_boost_round", 20)
    return b


def fake_cv(params, train_set, **kwargs):
    m = resolved_metric(params)
    return {m + "-mean": [synth(params)], m + "-stdv": [0.0]}


_opt.lgb.train = fake_train
_opt.lgb.cv = fake_cv


def run(cv):
    import numpy as np
    dtrain = lgb.Dataset(np.zeros((10, 3)), label=np.zeros(10))
    params = {"objective": "regression", "metric": "l2", "verbosity": -1}
    if cv:
        tuner = lgb.LightGBMTunerCV(params, dtrain, num_boost_round=20, optuna_seed=42, show_progress_bar=False)
    else:
        dval = lgb.Dataset(np.zeros((4, 3)), label=np.zeros(4))
        tuner = lgb.LightGBMTuner(params, dtrain, valid_sets=[dval], num_boost_round=20,
                                  optuna_seed=42, show_progress_bar=False)
    tuner.run()
    st = tuner.study
    trials = [{
        "number": t.number,
        "step": t.system_attrs.get("lightgbm_tuner:step_name"),
        "params": t.params,
        "value": t.value,
    } for t in st.trials]
    return {
        "direction": int(st.direction),
        "n_trials": len(st.trials),
        "best_params": tuner.best_params,
        "best_score": tuner.best_score,
        "trials": trials,
    }


out = {"train": run(cv=False), "cv": run(cv=True)}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightgbm_tuner_synth.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path, "-", out["train"]["n_trials"], "train trials,", out["cv"]["n_trials"], "cv trials")
