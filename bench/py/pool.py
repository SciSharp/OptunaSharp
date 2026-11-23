"""Waiting-trials sentence of the Parallel-trials section, Python side (probes/pool/Program.cs mirrors it): Optuna or Rustuna.

Usage: python -I pyrun.py pool.py <optuna|rustuna>

640 trials sleeping 100 ms on 64 workers, then 1,280 on 128 (ideal 1.0 s each). Optuna runs the workers through
``n_jobs``; Rustuna through as many Python threads, each calling ``study.optimize`` with trials / workers trials. Each
configuration runs under benchlib's protocol (warm-up runs never counted, then measured runs, each a fresh study); until
2026-11-23 each ran once.
"""
import sys
import threading
import time

import benchlib

lib = sys.argv[1] if len(sys.argv) > 1 else "optuna"
if lib == "rustuna":
    import rustuna as L
else:
    import optuna as L

    L.logging.set_verbosity(L.logging.WARNING)


def one(jobs, trials):
    """One run: a fresh study of ``trials`` sleeping trials on ``jobs`` workers, counting how many ran at once.

    Args:
        jobs: The worker count.
        trials: The trial count.

    Returns:
        ``(seconds, fingerprint, extra)`` for benchlib; the fingerprint is the trial count only (with ``n_jobs > 1``
        Optuna reseeds the sampler's generator per worker, so the best value varies while the work, a 100 ms sleep, does
        not), and the extra fields carry the ideal time and the peak concurrency, the proof that the workers overlapped.
    """
    state = {"live": 0, "peak": 0}
    lock = threading.Lock()

    def obj(t):
        """Sleep 100 ms (a trial that waits rather than computes), counting how many run at once."""
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        x = t.suggest_float("x", -1, 1)
        time.sleep(0.1)
        with lock:
            state["live"] -= 1
        return x * x

    s = L.create_study(sampler=L.samplers.RandomSampler(seed=0))
    t0 = time.perf_counter()
    if lib == "optuna":
        s.optimize(obj, n_trials=trials, n_jobs=jobs)
    else:
        workers = [threading.Thread(target=s.optimize, args=(obj,), kwargs={"n_trials": trials // jobs})
                   for _ in range(jobs)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    dt = time.perf_counter() - t0
    ideal = -(-trials // jobs) * 0.1
    return (dt, str(len(s.trials)),
            f"jobs={jobs} trials={trials} ideal={ideal:.2f}s peakConcurrent={state['peak']}")


tag = "Rustuna" if lib == "rustuna" else "Python"
for jobs, trials in [(64, 640), (128, 1280)]:
    runs = benchlib.run(f"jobs{jobs}", lambda jobs=jobs, trials=trials: one(jobs, trials))
    print(f"{tag}: jobs={jobs} trials={trials} median={benchlib.median(runs):.3f}s ideal={-(-trials // jobs) * 0.1:.2f}s",
          flush=True)
