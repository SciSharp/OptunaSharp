"""Large-studies table, Python side (probes/proof/Program.cs mirrors it): the workloads of Rustuna's announcement.

Usage: python -I pyrun.py pbench.py <rustuna|optuna> <tpe|motpe|cmaes|nsga2> <trials> <threads> <warmup_trials>

Workloads, identical in proof/Program.cs: 40 float parameters x0..x39 in [-10, 10], seed 0, default sampler
settings. Single-objective: sum((x - 1)^2). Two objectives (MOTPE, NSGA-II): sum(x^2) and sum((x - 2)^2).
Threads: Optuna uses n_jobs; Rustuna has no n_jobs, so T Python threads each run study.optimize on the same
study with trials / T trials (its samplers release the GIL while sampling). Under benchlib's protocol since 2026-11-23:
warm-up runs of the same study at ``warmup_trials`` trials (a tenth of the measured size; it imports and runs the same
code, and Optuna's 1,000-trial studies take 38-197 s, so full-size warm-up runs would triple the cost), never counted;
then measured runs of the full study, each a new study. Only optimize() is timed. Until 2026-11-23 the README took the
first run in a fresh process.

For Optuna runs the fingerprint is the trial hash: the first 16 hex digits of a SHA-256 over every trial's number,
state, parameters (name and IEEE-754 bits) and values, built byte for byte like proof/Program.cs's Hash. Equal hashes
on the same machine prove OptunaSharp ran Optuna's exact trials. Rustuna draws its own random numbers, so its trials
differ from the first one; its fingerprint is the trial count and best value.
"""
import hashlib
import struct
import sys
import threading
import time

import benchlib

lib, cfg = sys.argv[1], sys.argv[2]
trials, threads, warmup_trials = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

if lib == "rustuna":
    import rustuna as L
else:
    import optuna as L

    # Optuna logs every trial at INFO by default; Rustuna and OptunaSharp (Error level) print nothing.
    L.logging.set_verbosity(L.logging.WARNING)

S = L.samplers
NAMES = [f"x{i}" for i in range(40)]  # built once, as the C# side does


def single(trial):
    """sum((x - 1)^2) over the 40 parameters."""
    s = 0.0
    for name in NAMES:
        v = trial.suggest_float(name, -10, 10)
        s += (v - 1) * (v - 1)
    return s


def multi(trial):
    """(sum(x^2), sum((x - 2)^2)) over the 40 parameters."""
    a = b = 0.0
    for name in NAMES:
        v = trial.suggest_float(name, -10, 10)
        a += v * v
        b += (v - 2) * (v - 2)
    return a, b


def study_hash(study) -> str:
    """The C# probe's trial hash: number and state as little-endian int32, each parameter as its UTF-8 name, a zero
    byte and its float's IEEE-754 bits (little-endian int64), then each value's bits, in trial and suggestion order.

    Args:
        study: A finished Optuna study.

    Returns:
        The first 16 upper-case hex digits of the SHA-256, the format proof/Program.cs prints.
    """
    sha = hashlib.sha256()
    for t in study.trials:
        buf = bytearray(struct.pack("<ii", t.number, int(t.state.value)))
        for name, value in t.params.items():
            buf += name.encode("utf-8") + b"\x00" + struct.pack("<d", float(value))
        for v in t.values or ():
            buf += struct.pack("<d", float(v))
        sha.update(buf)
    return sha.hexdigest()[:16].upper()


multi_objective = cfg in ("motpe", "nsga2")
sampler_type = {"tpe": "TPESampler", "motpe": "TPESampler", "cmaes": "CmaEsSampler", "nsga2": "NSGAIISampler"}[cfg]


def one(n):
    """One study of ``n`` trials: built untimed, ``optimize`` timed, then fingerprinted.

    Args:
        n: The trial count (the warm-up size or the measured size).

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib.
    """
    kwargs = {"directions": ["minimize", "minimize"]} if multi_objective else {}
    study = L.create_study(sampler=getattr(S, sampler_type)(seed=0), **kwargs)
    objective = multi if multi_objective else single
    t0 = time.perf_counter()
    if threads == 1:
        study.optimize(objective, n_trials=n)
    elif lib == "optuna":
        study.optimize(objective, n_trials=n, n_jobs=threads)
    else:
        workers = [threading.Thread(target=study.optimize, args=(objective,), kwargs={"n_trials": n // threads})
                   for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    dt = time.perf_counter() - t0
    if lib == "optuna" and threads == 1:
        fingerprint = study_hash(study)
    else:
        # Rustuna's trial objects follow Optuna's API, but the old probe never read their values; a fingerprint is a check,
        # not the workload, so if the values cannot be read the trial count alone stands in rather than failing the run.
        try:
            best = min(t.values[0] for t in study.trials if t.values) if multi_objective else study.best_trial.value
            fingerprint = f"{len(study.trials)}:{benchlib.fp(best)}"
        except (AttributeError, TypeError, ValueError):
            fingerprint = str(len(study.trials))
    return dt, fingerprint, f"trials={n} threads={threads} completed={len(study.trials)}"


benchlib.run(f"{cfg}-{trials}", lambda: one(warmup_trials), lambda: one(trials))
