"""Parallel-trials table, Python side (probes/njobs/Program.cs mirrors it): Optuna 5.0 or Rustuna.

Usage: python -I pyrun.py njobs_scaling.py <optuna|rustuna>

The same ~50 ms CPU-bound objective as the C# probe (pure Python, so it holds the GIL), RandomSampler, 64 trials,
1/2/4/8/16 workers. Each worker count runs under benchlib's protocol (warm-up runs never counted, then measured runs, each
a fresh study); until 2026-11-23 it was three runs per worker count with no warm-up, and the median. Optuna runs its
workers through ``n_jobs`` (a thread pool); Rustuna has no ``n_jobs``, so N Python threads each call ``study.optimize``
on the same study with 64 / N trials, the way Rustuna's own documentation runs parallel trials. The loop length is
calibrated to ~50 ms in this process.
"""
import math
import os
import sys
import threading
import time

import benchlib

lib = sys.argv[1] if len(sys.argv) > 1 else "optuna"
if lib == "rustuna":
    import rustuna as L

    version = __import__("importlib.metadata").metadata.version("rustuna")
    label = f"Rustuna {version}"
else:
    import optuna as L

    L.logging.set_verbosity(L.logging.WARNING)
    label = f"Optuna {L.__version__}"


def burn(x, iters):
    """``iters`` sines: about 50 ms of single-threaded floating-point work once calibrated."""
    s = 0.0
    for i in range(iters):
        s += math.sin(x + i * 1e-3)
    return s


# Calibrate in this process, after two warm-up calls, so the 1-worker time is ~64 x 50 ms = 3.2 s; the parser's validity
# rule rejects a process whose calibration missed by more than 10 %.
iters = 200_000
for _ in range(2):
    burn(0.5, iters)
t = time.perf_counter()
burn(0.5, iters)
iters = int(iters * 50.0 / ((time.perf_counter() - t) * 1000.0))
N_TRIALS = 64
cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
print(f"Python {sys.version.split()[0]} + {label} | {cpus} logical CPUs | "
      f"{N_TRIALS} trials x ~50 ms CPU-bound objective | iters={iters}")


def objective(trial):
    """(x - 2)^2 plus a burn whose result keeps the loop alive without changing the value."""
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2 + burn(x, iters) * 1e-300


def one(n_jobs):
    """One run: a fresh 64-trial study at ``n_jobs`` workers, ``optimize`` (or the threads) timed.

    Args:
        n_jobs: The worker count.

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib; the fingerprint is the trial count only: with ``n_jobs > 1``
        Optuna reseeds the sampler's generator per worker (``reseed_rng``), so the best value varies between runs, while
        the work does not (every trial burns the same ~50 ms whatever x it drew).

    Raises:
        AssertionError: The study did not end with 64 trials (a lost or extra trial would change the work).
    """
    study = L.create_study(sampler=L.samplers.RandomSampler(seed=1))
    t0 = time.perf_counter()
    if n_jobs == 1:
        study.optimize(objective, n_trials=N_TRIALS)
    elif lib == "optuna":
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=n_jobs)
    else:
        workers = [threading.Thread(target=study.optimize, args=(objective,),
                                    kwargs={"n_trials": N_TRIALS // n_jobs}) for _ in range(n_jobs)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    dt = time.perf_counter() - t0
    assert len(study.trials) == N_TRIALS, len(study.trials)
    return dt, str(len(study.trials)), f"njobs={n_jobs}"


for n_jobs in (1, 2, 4, 8, 16):
    runs = benchlib.run(f"njobs{n_jobs}", lambda n_jobs=n_jobs: one(n_jobs))
    print(f"n_jobs={n_jobs:2}  median {benchlib.median(runs):7.3f} s over {len(runs)} measured runs", flush=True)
