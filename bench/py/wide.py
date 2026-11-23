"""High-dimensional rows of the Samplers table, Python side (probes/wide/Program.cs mirrors it): Optuna 5.0 or Rustuna.

Usage: python -I pyrun.py wide.py <optuna|rustuna> <tpe_ind|tpe_mv|cmaes|gp|random>

Configs: TPE independent 100-D x 150 trials, TPE multivariate 100-D x 150, CMA-ES 200-D x 100, GP 30-D x 40 and
Random 1000-D x 200, each minimizing sum((x_i - 1)^2) over floats in [-5, 5]. Under benchlib's protocol (warm-up runs
never counted, then measured runs, each a fresh sampler and study with the same seed, timing only ``optimize``); until
2026-11-23 each config ran once in a fresh process. Rustuna 0.3.1 has no GP sampler: that config prints a ``skip`` line.
"""
import sys
import time
import warnings

import benchlib

lib, cfg = sys.argv[1], sys.argv[2]
if lib == "rustuna":
    import rustuna as L
else:
    import optuna as L

    L.logging.set_verbosity(L.logging.ERROR)

warnings.simplefilter("ignore")
S = L.samplers
name, kwargs, dims, trials = {
    "tpe_ind": ("TPESampler", {"multivariate": False}, 100, 150),
    "tpe_mv": ("TPESampler", {"multivariate": True}, 100, 150),
    "cmaes": ("CmaEsSampler", {}, 200, 100),
    "gp": ("GPSampler", {}, 30, 40),
    "random": ("RandomSampler", {}, 1000, 200),
}[cfg]
tag = "Ru" if lib == "rustuna" else "Py"
if not hasattr(S, name):
    print(f"{tag} {cfg}: skip ({lib} has no {name})")
    sys.exit(0)


def obj(t):
    """sum((x_i - 1)^2) over ``dims`` floats in [-5, 5].

    The names are formatted on every call, as the C# probe's ``$"x{i}"`` is, so both sides do the same work.
    """
    s = 0.0
    for i in range(dims):
        v = t.suggest_float(f"x{i}", -5, 5)
        s += (v - 1) * (v - 1)
    return s


def one():
    """One run: a new sampler and study with seed 0, ``optimize`` timed.

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib: the best value is the fingerprint, equal for every run.
    """
    study = L.create_study(sampler=getattr(S, name)(seed=0, **kwargs))
    t0 = time.perf_counter()
    study.optimize(obj, n_trials=trials)
    dt = time.perf_counter() - t0
    return dt, benchlib.fp(study.best_trial.value), f"dims={dims} trials={trials}"


runs = benchlib.run(cfg, one)
print(f"{tag} {cfg} {dims}-D x {trials}: median {benchlib.median(runs):.3f}s over {len(runs)} measured runs")
