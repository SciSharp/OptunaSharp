"""Generate the CMA-ES optimizer-state oracle: the exact `cma:optimizer:*` pickles Optuna stores.

Optuna's CmaEsSampler keeps its cmaes optimizer (CMA, SepCMA or CMAwM) in trial system attrs as
`pickle.dumps(optimizer).hex()`, split into 2045-character chunks. OptunaSharp writes and reads the same
pickles, so C# and Python workers can share a CMA-ES study (review item P3-2). This script runs real Optuna
studies and records the stored pickles, so the C# reader and writer can be checked byte for byte with no
Python at test time (Samplers/CmaPickleOracleTests):

- the first pickle of a study comes from a fresh optimizer (cmaes' `_C` sits in its `__init__` position, so
  `_c_1d` follows `_mean`);
- every later one comes from an optimizer Optuna restored and then told (`__setstate__` re-adds `_C` last, so
  `_c_1d` is last, and the arrays the tell did not replace keep the dtype object unpickling gave them).

`codec_cases` are large pickles of an untold 125-dimensional cmaes.CMA, gzip-compressed and base64-encoded
(mostly zeros, so they stay small). They cross the 64 KiB protocol-4 frame target in both ways the pickler
handles: a frame committed at an object boundary once it passes the target, and a bytes object of at least
the target written outside any frame.

`protocol5_cases` are restored states re-pickled with protocol 5, the default from CPython 3.14, whose numpy
arrays are `_frombuffer` + BYTEARRAY8 instead of `_reconstruct` + BUILD.

Run in the repo's main Python env (the interpreter the interop suite uses; cmaes 0.13.1, numpy 2.4.2):
    python test/OptunaSharp.Tests/oracle/gen_cma_pickle_oracle.py
"""
import base64
import gzip
import json
import math
import os
import pickle

import cmaes
import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def mixed(t):
    """Two linear floats and one log-scaled float: the default CMA case.

    Args:
        t: The running trial.

    Returns:
        A smooth bowl with its minimum inside the box.
    """
    x = t.suggest_float("x", -5, 5)
    y = t.suggest_float("y", -5, 5)
    z = t.suggest_float("z", 1e-3, 10, log=True)
    return (x - 1) ** 2 + (y + 0.5) ** 2 + math.log(z) ** 2


def plane2(t):
    """Two linear floats.

    Args:
        t: The running trial.

    Returns:
        A quadratic bowl centered at (1, -0.5).
    """
    x = t.suggest_float("x", -5, 5)
    y = t.suggest_float("y", -5, 5)
    return (x - 1) ** 2 + (y + 0.5) ** 2


def one_d(t):
    """One float, for the single-dimensional separable-CMA fallback.

    Args:
        t: The running trial.

    Returns:
        A one-dimensional quadratic.
    """
    return (t.suggest_float("x", -5, 5) - 1) ** 2


def margin(t):
    """A float, an integer and a stepped float, so CMAwM has two discrete dimensions.

    Args:
        t: The running trial.

    Returns:
        A quadratic over the mixed space.
    """
    x = t.suggest_float("x", -5, 5)
    n = t.suggest_int("n", -3, 3)
    s = t.suggest_float("s", 0, 2, step=0.5)
    return (x - 1) ** 2 + (n - 1) ** 2 + (s - 1.5) ** 2


def sphere(dim):
    """Builds a shifted sphere over `dim` floats.

    Args:
        dim: The number of parameters.

    Returns:
        The objective function.
    """
    def obj(t):
        """The shifted sphere.

        Args:
            t: The running trial.

        Returns:
            The sum of squared distances from 0.5.
        """
        return sum((t.suggest_float(f"x{i}", -5, 5) - 0.5) ** 2 for i in range(dim))
    return obj


def sometimes_infinite(t):
    """Two floats, and an infinite value every fifth trial, so the funhist stores infinities.

    Args:
        t: The running trial.

    Returns:
        A quadratic, or infinity.
    """
    x = t.suggest_float("x", -5, 5)
    y = t.suggest_float("y", -5, 5)
    return math.inf if t.number % 5 == 3 else (x - 1) ** 2 + y ** 2


def warm_start_sources():
    """Twelve completed trials of `plane2`, drawn from a fixed seed, as warm-start sources.

    Returns:
        The source trials.
    """
    rng = np.random.RandomState(3)
    trials = []
    for _ in range(12):
        x, y = rng.uniform(-5, 5), rng.uniform(-5, 5)
        trials.append(create_trial(
            params={"x": x, "y": y},
            distributions={"x": FloatDistribution(-5, 5), "y": FloatDistribution(-5, 5)},
            value=(x - 1) ** 2 + (y + 0.5) ** 2))
    return trials


# name -> (sampler factory, objective, trials, prefix, how many stored pickles to keep)
VARIANTS = {
    "cma": (lambda: optuna.samplers.CmaEsSampler(seed=7), mixed, 40, "cma:", 4),
    "cma_lr_adapt": (lambda: optuna.samplers.CmaEsSampler(seed=7, lr_adapt=True), mixed, 40, "cma:", 4),
    "sepcma": (lambda: optuna.samplers.CmaEsSampler(seed=7, use_separable_cma=True), plane2, 40, "sepcma:", 4),
    # One dimension: Optuna ignores use_separable_cma and stores a CMA under the sepcma: prefix.
    "sepcma_1d": (lambda: optuna.samplers.CmaEsSampler(seed=7, use_separable_cma=True), one_d, 30, "sepcma:", 3),
    "cmawm": (lambda: optuna.samplers.CmaEsSampler(seed=7, with_margin=True), margin, 40, "cmawm:", 4),
    # with_margin on a continuous space: CMAwM without its margin attributes (n_zdim == 0).
    "cmawm_continuous": (lambda: optuna.samplers.CmaEsSampler(seed=7, with_margin=True), plane2, 30, "cmawm:", 3),
    # A user sigma0 is a Python float, so cmaes' _tolx (1e-12 * sigma) is a float, not numpy.float64.
    "cma_user_sigma0": (lambda: optuna.samplers.CmaEsSampler(seed=7, popsize=9, sigma0=0.7, x0={"x": 1.0, "y": -1.0}),
                        plane2, 40, "cma:", 3),
    # Warm start: get_warm_start_mgd's sigma is math.pow (a float) and it supplies the covariance.
    "cma_warm_start": (lambda: optuna.samplers.CmaEsSampler(seed=7, source_trials=warm_start_sources()),
                       plane2, 30, "cma:", 3),
    "cma_infinite": (lambda: optuna.samplers.CmaEsSampler(seed=7), sometimes_infinite, 40, "cma:", 3),
    # 30 dimensions: arrays past 255 bytes (BINBYTES) and a funhist past 255 entries (BININT2 shapes).
    "cma_30d": (lambda: optuna.samplers.CmaEsSampler(seed=7), sphere(30), 45, "cma:", 2),
}


def codec_cases():
    """Large pickles of an untold 125-dimensional cmaes.CMA, for the frame handling of the C# pickler.

    The fresh layout puts `_c_1d` (63,000 bytes) early, so the frame passes 64 KiB at a later object boundary
    and is committed there; `_ESigma` (125,000 bytes) is past the target and is written unframed. The funhist,
    which cmaes leaves uninitialized, is zeroed so the pickle is deterministic.

    Returns:
        Case name to gzip-compressed, base64-encoded pickle bytes.
    """
    cma = cmaes.CMA(mean=np.zeros(125), sigma=0.5, seed=1)
    cma._funhist_values = np.zeros_like(cma._funhist_values)
    raw = pickle.dumps(cma)
    return {"cma_125d_untold": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")}


def protocol5_cases(variants):
    """Restored states re-pickled with protocol 5, the form a Python 3.14 worker stores.

    CPython 3.14 made protocol 5 pickle's default, and numpy pickles a contiguous array there as
    `_frombuffer(bytearray, dtype, shape, order)` instead of `_reconstruct` + BUILD. Loading a state and
    dumping it again keeps every value and Python type, so OptunaSharp must restore each case to exactly the
    optimizer the protocol-4 state holds.

    Args:
        variants: The oracle's variants, as main() collected them.

    Returns:
        A list of {variant, index, hex}: the protocol-4 state each case re-pickles, and the protocol-5 hex.
    """
    cases = []
    for name in ("cma", "cma_lr_adapt", "sepcma", "cmawm", "cmawm_continuous"):
        state = variants[name]["pickles"][1]["hex"]
        p5 = pickle.dumps(pickle.loads(bytes.fromhex(state)), protocol=5)
        assert p5[:2] == b"\x80\x05", name
        cases.append({"variant": name, "index": 1, "hex": p5.hex()})
    return cases


def main():
    """Runs every variant and writes cma_pickle_oracle.json next to this script."""
    oracle = {
        "generator": "test/OptunaSharp.Tests/oracle/gen_cma_pickle_oracle.py",
        "optuna": optuna.__version__,
        "cmaes": cmaes.__version__,
        "numpy": np.__version__,
        "variants": {},
        "codec_cases": codec_cases(),
    }
    for name, (make_sampler, objective, n_trials, prefix, keep) in VARIANTS.items():
        study = optuna.create_study(sampler=make_sampler())
        study.optimize(objective, n_trials=n_trials)
        pickles = []
        for t in study.trials:
            attrs = {k: v for k, v in t.system_attrs.items() if k.startswith(prefix + "optimizer:")}
            if attrs:
                hexstr = "".join(attrs[f"{prefix}optimizer:{i}"] for i in range(len(attrs)))
                pickles.append({"trial": t.number, "chunks": len(attrs), "hex": hexstr})
        assert len(pickles) >= 2, f"{name}: needs a fresh and a restored pickle, got {len(pickles)}"
        oracle["variants"][name] = {"prefix": prefix, "pickles": pickles[:keep]}
        print(f"{name}: {len(pickles)} stored, kept {min(keep, len(pickles))}, "
              f"sizes {[len(p['hex']) // 2 for p in pickles[:keep]]}")
    oracle["protocol5_cases"] = protocol5_cases(oracle["variants"])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cma_pickle_oracle.json")
    # LF on every OS, as the repository stores text (.gitattributes eol=lf).
    with open(path, "w", newline="
") as f:
        json.dump(oracle, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
