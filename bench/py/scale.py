"""The 50,000-trial Random row of the Samplers table, Python side (probes/scale/Program.cs mirrors it): Optuna or Rustuna.

Usage: python -I pyrun.py scale.py <optuna|rustuna> inmem

One run is a whole 50,000-trial RandomSampler study on four floats in [-5, 5], timed per block of 10,000 to show whether
the per-trial cost stays flat as the in-memory history grows; the run's seconds are the whole study, the blocks are printed
as ``blocks=``. Under benchlib's protocol (warm-up runs never counted, then measured runs); until 2026-11-23 the study ran
once in a fresh process, where OptunaSharp's falling blocks were its JIT warming up. After the runs: the cost of one
``study.trials`` copy (inside each run's extra) and the process's resident set size.
"""
import sys
import time
import warnings

import benchlib

lib, mode = sys.argv[1], sys.argv[2]
if lib == "rustuna":
    import rustuna as L
else:
    import optuna as L

    L.logging.set_verbosity(L.logging.ERROR)

warnings.simplefilter("ignore")
if mode != "inmem":
    raise SystemExit("only the inmem mode is part of the README run")


def obj(t):
    """sum(x_i^2) over four floats in [-5, 5]."""
    s = 0.0
    for i in range(4):
        v = t.suggest_float(f"x{i}", -5, 5)
        s += v * v
    return s


def one():
    """One run: a new 50,000-trial study timed per block of 10,000.

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib: the whole study's seconds, the trial count and best value as the
        fingerprint, and the five block times and one trials-copy cost as extra fields.
    """
    study = L.create_study(sampler=L.samplers.RandomSampler(seed=0))
    blocks, total = [], 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        study.optimize(obj, n_trials=10_000)
        dt = time.perf_counter() - t0
        total += dt
        blocks.append(f"{dt:.6f}")
    # Not part of the run's seconds: one copy of the 50,000 trials, the cost a caller pays for study.trials.
    t0 = time.perf_counter()
    n = len(study.trials)
    copy_ms = (time.perf_counter() - t0) * 1000
    return total, f"{n}:{benchlib.fp(study.best_trial.value)}", f"blocks={','.join(blocks)} trials-copy={copy_ms:.0f}ms"


benchlib.run("inmem", one)
import psutil  # noqa: E402 - only for the memory number, after every timed run

rss = psutil.Process().memory_info().rss // 1048576
tag = "Ru" if lib == "rustuna" else "Py"
print(f"{tag} inmem 50k: rss={rss}MB")
