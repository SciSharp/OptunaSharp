"""The measurement protocol every Python probe shares (bench/README.md, "Protocol"); common/Bench.cs is its C# twin.

Each process first repeats the workload until it is warm and discards those runs, then measures. The rules and the printed
line format are the C# side's, so bench/parse.py reads Optuna, Rustuna and OptunaSharp alike:

    bench <label> <warmup|measure> <index> <seconds>s mods=+<n> fp=<fingerprint> [extra]
    bench <label> done warmup=<n> measure=<m> stop=<quiet|maxruns|maxseconds> quiet_tail=<k>

``mods`` is how many modules the run imported (``sys.modules`` grew by that much). Python has no JIT, so its first run's
extra cost is lazy imports (torch for the GP sampler, cmaes for CMA-ES, SciPy's QMC engine); the warm-up goes on until the
last quarter of its runs imported nothing (the C# side's quiet rule, applied to imports), and a measured run that still
imports something was not warm, which the parser reports, as it reports a .NET run that still compiled methods. The C#
side's other rule, at least 70 warm-up runs for .NET's two tiering steps, has no Python counterpart: CPython 3.12
specializes its bytecode within a few executions, all inside the first run of these workloads.

Why: until 2026-11-23 each workload was timed once, in a fresh process, so the README compared first runs, mostly start-up
costs (.NET's tiered JIT on one side, lazy imports on the other) rather than the libraries' steady-state speed.
"""
import gc
import math
import os
import sys
import time

_settings_printed = False


class Settings:
    """The warm-up and measuring budgets, from the ``BENCH_*`` environment variables the runner sets for every probe.

    Attributes:
        warmup_seconds: Warm up for at least this many seconds (``BENCH_WARMUP_SECONDS``, default 5).
        warmup_runs: ...and at least this many runs (``BENCH_WARMUP_RUNS``, default 2).
        quiet_share: ...then until the last ``ceil(quiet_share x warm-up runs)`` runs, at least the last one, imported
            nothing (``BENCH_WARMUP_QUIET_SHARE``, default 0.25; 0 turns the rule off).
        max_warmup_runs: Stop warming up after this many runs even if the runs still import (``BENCH_WARMUP_MAX_RUNS``,
            default 500)...
        max_warmup_seconds: ...or once this many seconds have passed with the first two minimums met
            (``BENCH_WARMUP_MAX_SECONDS``, default 60).
        measure_runs: Measure at least this many runs (``BENCH_MEASURE_RUNS``, default 3).
        measure_seconds: ...then keep measuring until this many measured seconds (``BENCH_MEASURE_SECONDS``, default 10)...
        max_measure_runs: ...or this many runs, whichever comes first (``BENCH_MAX_RUNS``, default 30; 10 before
            2026-11-23 10:45, raised so a short workload fills its measured seconds with more runs).
    """

    def __init__(self):
        """Read the settings from the environment.

        Raises:
            ValueError: A variable is set but is not a number, or the budgets are inconsistent (a minimum above its
                maximum, a negative count or time, a quiet share outside 0..1), which would silently measure fewer runs
                than the method promises.
        """
        self.warmup_seconds = _number("BENCH_WARMUP_SECONDS", 5.0)
        self.warmup_runs = int(_number("BENCH_WARMUP_RUNS", 2))
        self.quiet_share = _number("BENCH_WARMUP_QUIET_SHARE", 0.25)
        self.max_warmup_runs = int(_number("BENCH_WARMUP_MAX_RUNS", 500))
        self.max_warmup_seconds = _number("BENCH_WARMUP_MAX_SECONDS", 60.0)
        self.measure_runs = int(_number("BENCH_MEASURE_RUNS", 3))
        self.measure_seconds = _number("BENCH_MEASURE_SECONDS", 10.0)
        self.max_measure_runs = int(_number("BENCH_MAX_RUNS", 30))
        if (self.warmup_seconds < 0 or self.warmup_runs < 0 or self.measure_runs < 1 or self.measure_seconds < 0
                or self.warmup_runs > self.max_warmup_runs or self.warmup_seconds > self.max_warmup_seconds
                or self.measure_runs > self.max_measure_runs or not 0 <= self.quiet_share <= 1):
            raise ValueError(f"inconsistent BENCH_* settings: {vars(self)}")

    def quiet_tail(self, quiet):
        """Whether the warm-up's trailing runs were quiet enough to stop warming up.

        Args:
            quiet: One flag per warm-up run so far, in order: ``True`` when the run imported nothing.

        Returns:
            ``True`` when the last ``max(1, ceil(quiet_share x runs))`` runs were all quiet, or when the rule is off
            (``quiet_share`` 0); ``False`` with no runs yet, since nothing has shown the process quiet.
        """
        if self.quiet_share <= 0:
            return True
        need = max(1, math.ceil(self.quiet_share * len(quiet)))
        return len(quiet) >= need and all(quiet[-need:])


def _number(name, fallback):
    """One numeric environment variable.

    Args:
        name: The variable's name.
        fallback: The value when the variable is unset or empty.

    Returns:
        The value as a float.

    Raises:
        ValueError: The variable is set but is not a number.
    """
    text = os.environ.get(name, "").strip()
    if not text:
        return float(fallback)
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{name}={text} is not a number") from None


def fp(value):
    """The round-trip text of a float, the fingerprint form of a best value (equal text means equal bits).

    Args:
        value: The value.

    Returns:
        ``repr(float(value))``, the shortest text that reads back to the same float.
    """
    return repr(float(value))


def once(label, phase, index, body):
    """Run a workload once from a collected heap and print its line.

    Args:
        label: The workload's name within this process (no spaces).
        phase: ``warmup`` or ``measure``.
        index: The run's index within its phase.
        body: A callable returning ``(seconds, fingerprint, extra)``: the seconds of its own measured part (``optimize``
            only), a token that is equal for runs that did the same work, and further ``key=value`` text or ``""``.

    Returns:
        What ``body`` returned.
    """
    result, _ = _once(label, phase, index, body)
    return result


def _once(label, phase, index, body):
    """``once``, also returning how many modules the run imported (the warm-up's quiet rule needs it).

    Args:
        label: The workload's name within this process (no spaces).
        phase: ``warmup`` or ``measure``.
        index: The run's index within its phase.
        body: The run (see ``once``).

    Returns:
        ``((seconds, fingerprint, extra), imported)``.
    """
    # Outside the timer: the previous run's garbage, so this run starts from the same heap state as the C# side's.
    gc.collect()
    modules0 = len(sys.modules)
    seconds, fingerprint, extra = body()
    imported = len(sys.modules) - modules0
    tail = f" {extra}" if extra else ""
    print(f"bench {label} {phase} {index} {seconds:.6f}s mods=+{imported} fp={fingerprint}{tail}", flush=True)
    return (seconds, fingerprint, extra), imported


def run(label, warmup, measure=None, settings=None):
    """Warm up, then measure; every run is printed, the warm-up runs are never counted by the parser.

    Args:
        label: The workload's name within this process; no spaces (the parser splits on them).
        warmup: One warm-up run (see ``once`` for its return value). For long studies it may be the same workload with
            fewer trials: the same sampler, objective and seed import and run the same code, only less of it.
        measure: One measured run, or ``None`` to measure ``warmup`` itself.
        settings: The budgets, or ``None`` to read them from the environment.

    Returns:
        The measured runs' ``(seconds, fingerprint, extra)`` tuples, in order (never the warm-up runs), for a probe that
        prints its own summary line.

    Raises:
        ValueError: The label contains white space, or the environment's settings are invalid.
    """
    global _settings_printed
    if any(c.isspace() for c in label):
        raise ValueError(f"label {label!r} contains white space")
    s = settings or Settings()
    measure = measure or warmup
    if not _settings_printed:
        _settings_printed = True
        print(f"bench settings warmup>={s.warmup_seconds:g}s,{s.warmup_runs}runs,quiet{s.quiet_share:g}"
              f"(max{s.max_warmup_runs}runs,{s.max_warmup_seconds:g}s) "
              f"measure>={s.measure_runs}runs,until{s.measure_seconds:g}s(max{s.max_measure_runs})", flush=True)

    # Warm-up: at least warmup_runs runs AND warmup_seconds of wall time, then until the last quiet_share of the runs
    # imported nothing, unless max_warmup_runs or max_warmup_seconds is reached first (the first two minimums win over the
    # time cap, so a workload longer than the cap still gets its warmup_runs runs).
    start = time.perf_counter()
    quiet = []
    while True:
        elapsed = time.perf_counter() - start
        minimums = len(quiet) >= s.warmup_runs and elapsed >= s.warmup_seconds
        if minimums and s.quiet_tail(quiet):
            stop = "quiet"
            break
        if len(quiet) >= s.max_warmup_runs:
            stop = "maxruns"
            break
        if minimums and elapsed >= s.max_warmup_seconds:
            stop = "maxseconds"
            break
        _, imported = _once(label, "warmup", len(quiet), warmup)
        quiet.append(imported == 0)
    warm = len(quiet)
    quiet_tail = 0
    for q in reversed(quiet):
        if not q:
            break
        quiet_tail += 1

    # Measured: at least measure_runs runs, then on until measure_seconds of measured time or max_measure_runs runs.
    runs, total = [], 0.0
    while len(runs) < s.measure_runs or (total < s.measure_seconds and len(runs) < s.max_measure_runs):
        result = once(label, "measure", len(runs), measure)
        runs.append(result)
        total += result[0]
    print(f"bench {label} done warmup={warm} measure={len(runs)} stop={stop} quiet_tail={quiet_tail}", flush=True)
    return runs


def median(runs):
    """The median of measured runs' seconds, for a probe's own summary line.

    Args:
        runs: What ``run`` returned; at least one run.

    Returns:
        The median of the seconds (the mean of the two middle values for an even count).

    Raises:
        ValueError: ``runs`` is empty.

    Only for human-readable summaries; bench/parse.py recomputes every statistic from the per-run lines.
    """
    if not runs:
        raise ValueError("no runs")
    s = sorted(r[0] for r in runs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
