"""Samplers-table probe, Python side (probes/perf/Program.cs mirrors it): Optuna 5.0 or Rustuna, one config per process.

Usage: python -I pyrun.py perf.py <optuna|rustuna> <random|tpe|tpe_mv|gp|cmaes|nsga2|qmc>

For each sampler a seeded study minimizes sum((x_i - 1)^2) over six floats in [-5, 5] for a fixed trial count
(Random 5,000; TPE and TPE multivariate 500; GP 100; CMA-ES, NSGA-II and QMC 3,000), the same counts and objective as
the C# probe. Under benchlib's protocol (bench/README.md, "Protocol"): warm-up runs that are never counted, so the first
run's one-time costs (Python's lazy imports, such as torch for the GP sampler and cmaes for CMA-ES) are out of the
numbers as .NET's JIT warm-up is out of the C# side's; then measured runs, each a fresh sampler and study with the same
seed, timing only ``optimize``. Until 2026-11-23 each config ran once in a fresh process. Rustuna 0.3.1 has no GP or QMC
sampler: those configs print a ``skip`` line, which the parser reports as "not available".
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

    # Optuna logs every trial at INFO; Rustuna and OptunaSharp (Error level) print nothing.
    L.logging.set_verbosity(L.logging.ERROR)

# Optuna's QMC and GP samplers warn (experimental, independent sampling); the warnings are not part of the workload.
warnings.simplefilter("ignore")
S = L.samplers


def obj(t):
    """sum((x_i - 1)^2) over six floats in [-5, 5], the C# probe's objective."""
    s = 0.0
    for i in range(6):
        v = t.suggest_float(f"x{i}", -5, 5)
        s += (v - 1) * (v - 1)
    return s


picks = {
    "random": ("RandomSampler", {}, 5000),
    "tpe": ("TPESampler", {}, 500),
    "tpe_mv": ("TPESampler", {"multivariate": True}, 500),
    "gp": ("GPSampler", {}, 100),
    "cmaes": ("CmaEsSampler", {}, 3000),
    "nsga2": ("NSGAIISampler", {}, 3000),
    "qmc": ("QMCSampler", {}, 3000),
}
name, kwargs, n = picks[cfg]
tag = "Ru" if lib == "rustuna" else "Py"
if not hasattr(S, name):
    print(f"{tag} {cfg}: skip ({lib} has no {name})")
    sys.exit(0)


def one():
    """One run: a new sampler and study with seed 0 (a reused sampler would continue its random stream), ``optimize`` timed.

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib: the best value is the fingerprint, equal for every run.
    """
    study = L.create_study(sampler=getattr(S, name)(seed=0, **kwargs))
    t0 = time.perf_counter()
    study.optimize(obj, n_trials=n)
    dt = time.perf_counter() - t0
    return dt, benchlib.fp(study.best_trial.value), f"trials={n}"


runs = benchlib.run(cfg, one)
print(f"{tag} {cfg}: {n} trials, median {benchlib.median(runs):.3f}s over {len(runs)} measured runs")
