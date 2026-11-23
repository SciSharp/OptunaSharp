"""Summarize a benchmark run (bench/run_worker.sh) into the numbers README.md's Performance section prints, with their
confidence intervals and validity checks.

Usage: python -I parse.py <results.txt> [load.log] [--json summary.json] [--boot N] [--seed S]

The data: every workload ran in several fresh processes (the runner's rounds); inside each, the probe printed its warm-up
runs and then its measured runs (bench/common/Bench.cs, bench/py/benchlib.py). Warm-up runs are never counted. For each
workload (a "cell": block, library, probe, configuration and label):

- Per process: the median of its measured runs (robust to one slow run: a stray interrupt, a collection).
- Estimate: the trimmed mean of the processes' medians, the fastest and the slowest process dropped (the owner's rule of
  2026-11-22, "remove the upper and lower bounds"; with fewer than five processes nothing is dropped and the cell is
  flagged).
- 95 % confidence interval: a hierarchical bootstrap (Davison & Hinkley 1997, sec. 3.8; the two-level design of Kalibera &
  Jones, "Rigorous Benchmarking in Reasonable Time", 2013): resample the processes with replacement, within each resample
  its measured runs, recompute the estimate; the 2.5th and 97.5th percentiles of --boot replicates (default 4000, seeded so
  the output is reproducible).
- Ratios (Optuna / OptunaSharp speedups): the ratio of estimates, its interval from the ratio of bootstrap replicates
  (independent samples). nJobs speedups pair each process's 1-worker and n-worker medians.

Validity checks, per cell (a failed check is printed as a flag and listed at the end):
- steady: the process was warm. For each process, the change across its measured runs along their Theil-Sen trend line
  (the median pairwise slope, over the span, relative to the median run); the median of that over processes must lie
  within 3 % ("drift" otherwise).
- jit (.NET): compilation time during measured runs over their run time must stay under 1 % ("jit"); mods (Python): a
  measured run must import nothing ("imports").
- fingerprint: every measured run of a seeded workload printed the same fingerprint (best value or trial hash), so every run
  did the same work ("fingerprint").
- precision: the interval's half-width over the estimate ("ci" when above 5 %).
- processes: at least five ("processes").
Process checks: an exit code other than 0 or a CPU affinity other than the runner's drops the process; the parallel-trials
rule drops a sweep whose 1-worker median is not within 10 % of 64 x 50 ms. With load.log, the load guard: a process during
any of whose 2-s windows the guests, the host services or the other host sessions reached the runner's limits (its
"## guard" line; 50 % of one CPU each by default) is dropped, as is one with no window at all (unmonitored). The runner
applies the same rule as it goes and redoes such a process ("@@ void" lines, after which the parser forgets the voided
attempt's output), so a drop here means its attempts ran out; the report lists the voided attempts and the load peaks of
the counted processes.
"""
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

TAG = re.compile(r"^(\S+) (cs|py|ru) (\S+) (\S+) run(\d+) \| (.*)$")
START = re.compile(r"^@@ start (\S+) (cs|py|ru) (\S+) (\S+) run(\d+) ([\d.]+)$")
END = re.compile(r"^@@ end (\S+) (cs|py|ru) (\S+) (\S+) run(\d+) ([\d.]+) rc=(\d+)$")
VOID = re.compile(r"^@@ void (\S+) (cs|py|ru) (\S+) (\S+) run(\d+) (.*)$")
GUARD = re.compile(r"^## guard \| guests<([\d.]+) services<([\d.]+) others<([\d.]+) interval=([\d.]+)")
LOADLINE = re.compile(r"^(\S+) ([\d.]+) total=([\d.]+) guests=([\d.]+) services=([\d.]+)(?: others=([\d.]+))? cpu=([\d,]+)")
RUNLINE = re.compile(r"^bench (\S+) (warmup|measure) (\d+) ([\d.]+)s(.*)$")
# The end of a workload's warm-up and measuring; stop= (why the warm-up ended) since 2026-11-23 10:45.
DONE = re.compile(r"^bench (\S+) done warmup=(\d+) measure=(\d+)(?: stop=(\S+))?")
SERIAL = "2-3"
PARALLEL = "0-19"
SERIAL_BLOCKS = {"B", "C", "D", "G"}
NAMES = {"cs": "OptunaSharp", "py": "Optuna", "ru": "Rustuna"}
DRIFT = 0.03
MAX_JIT_SHARE = 0.01
MAX_REL_HALF_WIDTH = 0.05
# The guard's limits when results.txt has no "## guard" line (runs before 2026-11-23 10:30): the runner's defaults.
DEFAULT_GUARD = {"guests": 50.0, "services": 50.0, "others": 50.0, "interval": 2.0}


def fields(text):
    """The ``key=value`` fields of a run line's tail.

    Args:
        text: Everything after the run's seconds.

    Returns:
        ``{key: value}`` with the values as text.
    """
    return dict(re.findall(r"(\S+?)=(\S+)", text))


def median(values):
    """The median of a non-empty list."""
    return statistics.median(values)


def trend(values):
    """The relative change across a sequence of measured runs along its Theil-Sen trend line.

    Args:
        values: One process's measured run times, in run order; at least two, all positive.

    Returns:
        The median of the pairwise slopes, times the span of the sequence, over the median value: about -0.05 for runs
        that got 5 % faster from first to last, 0 for steady ones. Robust to a minority of outliers (a slow first run of
        three still shows; one slow run in the middle of ten does not) and to alternation between two levels, whose
        pairwise slopes cancel (Sen 1968, "Estimates of the regression coefficient based on Kendall's tau").
    """
    n = len(values)
    slopes = [(values[j] - values[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return median(slopes) * (n - 1) / median(values)


def estimator(process_medians):
    """The cell's statistic: the trimmed mean of the processes' medians.

    Args:
        process_medians: One median per process.

    Returns:
        The mean without the lowest and the highest value when there are at least five, else the plain mean.
    """
    v = sorted(process_medians)
    core = v[1:-1] if len(v) >= 5 else v
    return statistics.fmean(core)


class Cell:
    """One workload's measured runs, grouped by process, and everything computed from them.

    Attributes:
        key: ``(block, lang, probe, cfg, label)``.
        procs: ``{process round: [(seconds, fields), ...]}`` of measured runs, in run order.
        warm: ``{process round: [(seconds, fields), ...]}`` of warm-up runs, in run order.
        stops: ``{process round: reason}``: why each process's warm-up ended (``quiet``: the runtime had settled;
            ``maxseconds`` / ``maxruns``: a cap; ``n/a`` from probes older than the reason).
    """

    def __init__(self, key):
        """Create an empty cell.

        Args:
            key: ``(block, lang, probe, cfg, label)``.
        """
        self.key = key
        self.procs = defaultdict(list)
        self.warm = defaultdict(list)
        self.stops = {}
        self._stats = None

    def values(self, field=None):
        """The measured values per process.

        Args:
            field: ``None`` for the run's seconds, else the name of a numeric field (units stripped: ``5.4MB`` -> 5.4).

        Returns:
            ``{process: [value, ...]}`` for the processes whose runs all carry the field.
        """
        out = {}
        for p, runs in self.procs.items():
            if field is None:
                out[p] = [s for s, _ in runs]
            else:
                vals = [_number(f.get(field)) for _, f in runs]
                if all(v is not None for v in vals):
                    out[p] = vals
        return out

    def stats(self, rng, boot):
        """Estimate, confidence interval, bootstrap replicates and validity checks of the run seconds (computed once).

        Args:
            rng: The seeded random generator shared by the whole parse (reproducible intervals).
            boot: Bootstrap replicates.

        Returns:
            A dict: est, lo, hi, rel (half-width over estimate), cv (of the process medians), P, N (min-max measured runs
            per process), steady, jit_share, max_jit_methods, max_mods, fingerprints, first (trimmed mean of each process's
            first warm-up run, the cold run, for reference only), warmups (min-max), reps (the replicates), flags.
        """
        if self._stats is None:
            self._stats = summarize(self.values(), rng, boot)
            s = self._stats
            # Steady state: each process's change across its measured runs along their robust trend line, then the median
            # over processes. Until 2026-11-23 13:40 this compared the medians of the later and the earlier half, which a
            # workload alternating between two times (OptunaSharp's QMC: 0.037 s and 0.0395 s, run by run, a collection
            # cycle) flags as a 5 % drift whenever the halves hold different counts of fast runs.
            ratios = [1 + trend([x for x, _ in runs]) for runs in self.procs.values() if len(runs) >= 2]
            s["steady"] = median(ratios) if ratios else None
            jit_ms = sum(_number(f.get("jit", "").split("/")[-1]) or 0 for runs in self.procs.values() for _, f in runs)
            run_ms = sum(x for runs in self.procs.values() for x, _ in runs) * 1000
            s["jit_share"] = jit_ms / run_ms if self.key[1] == "cs" and run_ms else None
            methods = [int(re.match(r"\+(\d+)", f.get("jit", "+0")).group(1)) for runs in self.procs.values() for _, f in runs
                       if re.match(r"\+(\d+)", f.get("jit", ""))]
            s["max_jit_methods"] = max(methods) if methods else None
            mods = [int(f["mods"].lstrip("+")) for runs in self.procs.values() for _, f in runs if "mods" in f]
            s["max_mods"] = max(mods) if mods else None
            s["fingerprints"] = sorted({f.get("fp", "") for runs in self.procs.values() for _, f in runs})
            # The counted processes' warm-ups only: the parallel-trials rule removes processes from procs, not from warm.
            counted_warm = [runs for p, runs in self.warm.items() if p in self.procs]
            firsts = [runs[0][0] for runs in counted_warm if runs]
            s["first"] = estimator(firsts) if firsts else None
            counts = [len(r) for r in counted_warm]
            s["warmups"] = (min(counts), max(counts)) if counts else (0, 0)
            # Only the counted processes' reasons (the parallel-trials rule may have removed some from procs).
            reasons = [self.stops[p] for p in self.procs if p in self.stops]
            s["stops"] = {r: reasons.count(r) for r in sorted(set(reasons))}
            flags = []
            if s["P"] < 5:
                flags.append("processes")
            if s["steady"] is not None and abs(s["steady"] - 1) > DRIFT:
                flags.append("drift")
            if s["jit_share"] is not None and s["jit_share"] > MAX_JIT_SHARE:
                flags.append("jit")
            if s["max_mods"]:
                flags.append("imports")
            if len(s["fingerprints"]) > 1:
                flags.append("fingerprint")
            if s["rel"] is not None and s["rel"] > MAX_REL_HALF_WIDTH:
                flags.append("ci")
            s["flags"] = flags
        return self._stats


def _number(text):
    """A field's number with its unit stripped (``5.4MB`` -> 5.4, ``0.003s`` -> 0.003), or None."""
    if text is None:
        return None
    m = re.match(r"^[+-]?[\d.]+(?:[eE][+-]?\d+)?", text)
    return float(m.group(0)) if m else None


def read_load(path):
    """The windows of a load log (bench/monitor.py's ``log`` mode).

    Args:
        path: load.log.

    Returns:
        ``[(clock, epoch, total, guests, services, others, per_cpu), ...]`` in file order; ``others`` is None on lines
        written before monitor.py logged it.

    Raises:
        OSError: The file cannot be read.
    """
    samples = []
    for line in open(path, encoding="ascii", errors="replace"):
        m = LOADLINE.match(line)
        if m:
            samples.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)),
                            float(m.group(6)) if m.group(6) is not None else None,
                            [int(x) for x in m.group(7).split(",")]))
    return samples


def overlapping(samples, start, end, interval):
    """The windows that overlap a process's span, the rule monitor.py's guard uses.

    Args:
        samples: ``read_load``'s windows.
        start: The span's start (epoch seconds).
        end: The span's end (epoch seconds).
        interval: The logger's window length.

    Returns:
        The windows ending after ``start`` that began (their end minus 1.05 intervals) before ``end``: one that only
        touches the span's edge counts, the conservative side.
    """
    return [s for s in samples if s[1] > start and s[1] - 1.05 * interval < end]


def load_peaks(inside):
    """The highest guests, services and other-sessions load over some windows.

    Args:
        inside: Windows from ``read_load``, at least one.

    Returns:
        ``{"guests": (value, clock), "services": (value, clock), "others": (value, clock) or None}``.
    """
    out = {}
    for name, index in (("guests", 3), ("services", 4), ("others", 5)):
        values = [s for s in inside if s[index] is not None]
        worst = max(values, key=lambda s: s[index]) if values else None
        out[name] = (worst[index], worst[0]) if worst else None
    return out


def summarize(per_process, rng, boot):
    """The estimate and its hierarchical-bootstrap interval for one set of per-process measured values.

    Args:
        per_process: ``{process: [value, ...]}``, every list non-empty.
        rng: The seeded random generator.
        boot: Bootstrap replicates.

    Returns:
        A dict: est, lo, hi, rel, cv, P, N (min, max), reps (the replicates, for ratio intervals); est is None when there are
        no processes.
    """
    procs = [v for v in per_process.values() if v]
    if not procs:
        return {"est": None, "lo": None, "hi": None, "rel": None, "cv": None, "P": 0, "N": (0, 0), "reps": []}
    meds = [median(v) for v in procs]
    est = estimator(meds)
    reps = []
    for _ in range(boot):
        pick = [procs[rng.randrange(len(procs))] for _ in procs]
        reps.append(estimator([median([v[rng.randrange(len(v))] for _ in v]) for v in pick]))
    reps.sort()
    lo, hi = reps[int(0.025 * boot)], reps[min(boot - 1, int(0.975 * boot))]
    # A coefficient of variation needs a non-zero mean: a latency that rounds to 0.0 ms everywhere (starve's p50) has none.
    mean = statistics.fmean(meds)
    cv = statistics.stdev(meds) / mean if len(meds) > 1 and mean else None
    return {"est": est, "lo": lo, "hi": hi, "rel": (hi - lo) / 2 / est if est else None, "cv": cv,
            "P": len(procs), "N": (min(map(len, procs)), max(map(len, procs))), "reps": reps}


def ratio(num, den):
    """The ratio of two independent cells' estimates and its bootstrap interval.

    Args:
        num: The numerator's stats (an Optuna time for a speedup).
        den: The denominator's stats (the OptunaSharp time).

    Returns:
        ``{"est", "lo", "hi"}``, or None when either side has no estimate.
    """
    if not num or not den or num["est"] is None or den["est"] is None:
        return None
    reps = sorted(a / b for a, b in zip(num["reps"], den["reps"]))
    n = len(reps)
    return {"est": num["est"] / den["est"], "lo": reps[int(0.025 * n)], "hi": reps[min(n - 1, int(0.975 * n))]}


def paired_ratio(num_cell, den_cell, rng, boot):
    """A ratio whose two sides come from the same processes (nJobs 1 over nJobs n in one sweep), paired by process.

    Args:
        num_cell: The numerator's cell (nJobs 1).
        den_cell: The denominator's cell (nJobs n).
        rng: The seeded random generator.
        boot: Bootstrap replicates.

    Returns:
        ``{"est", "lo", "hi"}`` over the processes both cells have, or None when they share none.
    """
    a, b = num_cell.values(), den_cell.values()
    common = sorted(set(a) & set(b))
    if not common:
        return None
    est = estimator([median(a[p]) / median(b[p]) for p in common])
    reps = []
    for _ in range(boot):
        pick = [common[rng.randrange(len(common))] for _ in common]
        reps.append(estimator([median([a[p][rng.randrange(len(a[p]))] for _ in a[p]]) /
                               median([b[p][rng.randrange(len(b[p]))] for _ in b[p]]) for p in pick]))
    reps.sort()
    return {"est": est, "lo": reps[int(0.025 * boot)], "hi": reps[min(boot - 1, int(0.975 * boot))]}


def fmt(s, unit="s", digits=3):
    """One cell for the summary: ``est unit [lo-hi] +-rel% (PxN, cv %, steady, first) flags``."""
    if not s or s.get("est") is None:
        return "-"
    digits = int(os.environ.get("PARSE_DIGITS", digits))
    text = f"{s['est']:.{digits}f}{unit} [{s['lo']:.{digits}f}-{s['hi']:.{digits}f}]"
    if s.get("rel") is not None:
        text += f" +-{s['rel'] * 100:.1f}%"
    if "P" in s:
        n = s["N"]
        text += f" ({s['P']}x{n[0] if n[0] == n[1] else f'{n[0]}-{n[1]}'}"
        if s.get("cv") is not None:
            text += f", cv {s['cv'] * 100:.1f}%"
        if s.get("steady") is not None:
            text += f", steady {s['steady']:.3f}"
        if s.get("first") is not None:
            text += f", first run {s['first']:.{digits}f}{unit}"
        if s.get("stops"):
            lo, hi = s.get("warmups", (0, 0))
            text += f", warm-up {lo if lo == hi else f'{lo}-{hi}'} runs, " + "/".join(f"{k} {v}" for k, v in s["stops"].items())
        text += ")"
    if s.get("flags"):
        text += " !" + ",".join(s["flags"])
    return text


def fmt_ratio(r, digits=2):
    """A ratio for the summary: ``est x [lo-hi]``."""
    if not r:
        return "-"
    return f"{r['est']:.{digits}f}x [{r['lo']:.{digits}f}-{r['hi']:.{digits}f}]"


def main():
    """Read the files named on the command line, print the summary and write the JSON."""
    args = sys.argv[1:]
    opts = {"--json": None, "--boot": "4000", "--seed": "20261123"}
    pos = []
    i = 0
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            pos.append(args[i])
            i += 1
    res_path = pos[0]
    load_path = pos[1] if len(pos) > 1 else None
    boot = int(opts["--boot"])
    rng = random.Random(int(opts["--seed"]))

    out = defaultdict(list)  # (block, lang, probe, cfg, run) -> lines
    spans = {}  # key -> [start, end, rc]
    header = []
    voided = []  # "key: verdict" of every attempt the runner redid
    guard = dict(DEFAULT_GUARD)
    for line in open(res_path, encoding="utf-8", errors="replace").read().splitlines():
        if line.startswith("##"):
            header.append(line)
            m = GUARD.match(line)
            if m:
                guard = {"guests": float(m.group(1)), "services": float(m.group(2)), "others": float(m.group(3)),
                         "interval": float(m.group(4))}
            continue
        m = START.match(line)
        if m:
            spans[m.groups()[:5]] = [float(m.group(6)), None, None]
            continue
        m = END.match(line)
        if m:
            key = m.groups()[:5]
            spans.setdefault(key, [None, None, None])[1:] = [float(m.group(6)), int(m.group(7))]
            continue
        m = VOID.match(line)
        if m:
            # The runner redoes this process under the same key: forget the voided attempt's lines and span, so only the
            # redo's (which follow) are counted.
            key = m.groups()[:5]
            out.pop(key, None)
            spans.pop(key, None)
            voided.append(f"{' '.join(key)}: {m.group(6)}")
            continue
        m = TAG.match(line)
        if m:
            out[m.groups()[:5]].append(m.group(6))

    samples = read_load(load_path) if load_path else []

    # Drop failed, mis-pinned and (with load.log) loaded or unmonitored processes.
    dropped = []
    good = {}
    for key, body in out.items():
        block = key[0]
        span = spans.get(key, [None, None, None])
        if span[2] not in (0, None):
            dropped.append(f"{' '.join(key)}: rc={span[2]}")
            continue
        env = next((b for b in body if b.startswith("probe env:")), "")
        aff = re.search(r"affinity=(\S+)", env)
        want = SERIAL if block in SERIAL_BLOCKS else PARALLEL
        if not aff or aff.group(1) != want:
            dropped.append(f"{' '.join(key)}: affinity {aff.group(1) if aff else 'missing'} (want {want})")
            continue
        if load_path:
            if span[0] is None or span[1] is None:
                dropped.append(f"{' '.join(key)}: no start or end line (the run stopped inside it)")
                continue
            inside = overlapping(samples, span[0], span[1], guard["interval"])
            if not inside:
                dropped.append(f"{' '.join(key)}: no load window (the logger was not running)")
                continue
            peaks = load_peaks(inside)
            over = [f"{name} {peaks[name][0]:.1f} at {peaks[name][1]} (limit {guard[name]:g})"
                    for name in ("guests", "services", "others")
                    if peaks[name] is not None and peaks[name][0] >= guard[name]]
            if over:
                dropped.append(f"{' '.join(key)}: load guard: " + ", ".join(over))
                continue
        good[key] = body

    # Cells from the run lines.
    cells = {}
    for (block, lang, probe, cfg, run), body in good.items():
        for x in body:
            m = RUNLINE.match(x)
            if not m:
                d = DONE.match(x)
                if d:
                    ckey = (block, lang, probe, cfg, d.group(1))
                    cells.setdefault(ckey, Cell(ckey)).stops[int(run)] = d.group(4) or "n/a"
                continue
            label, phase, _, seconds, tail = m.groups()
            ckey = (block, lang, probe, cfg, label)
            cell = cells.setdefault(ckey, Cell(ckey))
            (cell.procs if phase == "measure" else cell.warm)[int(run)].append((float(seconds), fields(tail)))

    def cell(block, lang, probe, cfg, label):
        """The cell's stats, or None when nothing was measured."""
        c = cells.get((block, lang, probe, cfg, label))
        return c.stats(rng, boot) if c else None

    print("=" * 120)
    for h in header:
        if h.startswith("## ") and " | idle:" not in h:
            print(h)
    gates = [h for h in header if " | idle:" in h]
    waited = [int(x) for h in gates for x in re.findall(r"waited (\d+)s", h)]
    rounds = sorted({int(k[4]) for k in good})
    print(f"## processes counted: {len(good)}; rounds: {rounds}; gates passed: {len(gates)}; longest gate wait: "
          f"{max(waited, default=0)} s; bootstrap replicates {boot}, seed {opts['--seed']}")
    if load_path:
        print(f"## load guard: every 2-s window of a counted process under guests {guard['guests']:g}, services "
              f"{guard['services']:g}, other sessions {guard['others']:g} (% of one CPU); attempts voided and redone by "
              f"the runner: {len(voided)}")
    else:
        print("## no load.log given: the load guard was not applied")
    # An abort (a gate timed out) is final unless a RESUME=1 start of the same run followed it ("## resume" after it in
    # the header lines, which keep the file's order).
    last_abort = max((i for i, h in enumerate(header) if "ABORTED" in h), default=None)
    if last_abort is not None:
        if any(h.startswith("## resume") for h in header[last_abort + 1:]):
            print("## the run aborted and was resumed (RESUME=1): its clean processes were kept, the rest ran again")
        else:
            print("## THE RUN ABORTED and was not resumed, so cells may lack processes: " + header[last_abort])
    if voided:
        print("## voided attempts (redone by the runner; their output is not counted):")
        for v in voided:
            print("   " + v)
    if dropped:
        print("## dropped processes:")
        for d in dropped:
            print("   " + d)

    summary = {"cells": {}, "derived": {}, "dropped": dropped, "voided": voided, "guard": guard, "header": header}

    def record(name, s):
        """Keep a cell's stats (without replicates) in the JSON."""
        if s:
            summary["cells"][name] = {k: v for k, v in s.items() if k != "reps"}

    # A - parallel trials. A process counts only if its own 1-worker calibration held (within 10 % of 64 x 50 ms).
    print("\n### A - Parallel trials (64 x ~50 ms CPU-bound, RandomSampler); per process: median of measured runs")
    for lang, probe in (("cs", "njobs"), ("py", "njobs_scaling"), ("ru", "njobs_scaling")):
        base = cells.get(("A", lang, probe, "sweep", "njobs1"))
        if not base:
            continue
        invalid = [p for p, v in base.values().items() if abs(median(v) - 3.2) / 3.2 > 0.10]
        print(f"  {NAMES[lang]}: processes {sorted(base.values())}, invalid (1-worker off by >10 %): {invalid or 'none'}")
        for nj in (1, 2, 4, 8, 16):
            c = cells.get(("A", lang, probe, "sweep", f"njobs{nj}"))
            if not c:
                continue
            for p in invalid:
                c.procs.pop(p, None)
            s = c.stats(rng, boot)
            record(f"A|{lang}|njobs{nj}", s)
            line = f"    nJobs {nj:2}: {fmt(s)}"
            if lang == "cs" and nj > 1:
                sp = paired_ratio(base, c, rng, boot)
                summary["derived"][f"A|cs|speedup{nj}"] = sp
                line += f" | speedup vs nJobs 1: {fmt_ratio(sp)}"
            print(line)

    # B - samplers.
    print("\n### B - Samplers (6 params unless stated)")
    rows = [("perf", "random", "Random, 6 parameters, 5,000 trials"), ("perf", "tpe", "TPE, 6 parameters, 500 trials"),
            ("perf", "tpe_mv", "TPE multivariate, 6 parameters, 500 trials"), ("perf", "gp", "GP, 6 parameters, 100 trials"),
            ("perf", "cmaes", "CMA-ES, 6 parameters, 3,000 trials"), ("perf", "nsga2", "NSGA-II, 6 parameters, 3,000 trials"),
            ("perf", "qmc", "QMC, 6 parameters, 3,000 trials"), ("wide", "random", "Random, 1,000 parameters, 200 trials"),
            ("wide", "tpe_ind", "TPE, 100 parameters, 150 trials"), ("wide", "cmaes", "CMA-ES, 200 parameters, 100 trials")]
    for probe, cfg, label in rows:
        print(f"  {label}:")
        got = {}
        for lang in ("cs", "py", "ru"):
            s = cell("B", lang, probe, cfg, cfg)
            got[lang] = s
            record(f"B|{lang}|{probe}|{cfg}", s)
            print(f"    {NAMES[lang]:11s} {fmt(s)}")
        sp = ratio(got["py"], got["cs"])
        summary["derived"][f"B|speedup|{probe}|{cfg}"] = sp
        print(f"    Optuna / OptunaSharp: {fmt_ratio(sp)}")
    print("  Random, 4 parameters, 50,000 trials, per block of 10,000 (each process: median per block over its measured runs):")
    blocks = {}
    for lang in ("cs", "py", "ru"):
        c = cells.get(("B", lang, "scale", "inmem", "inmem"))
        if not c:
            continue
        whole = c.stats(rng, boot)
        record(f"B|{lang}|scale|inmem", whole)
        per = []
        for b in range(5):
            bc = Cell(("B", lang, "scale", "inmem", f"block{b}"))
            for p, runs in c.procs.items():
                for _, f in runs:
                    bc.procs[p].append((float(f["blocks"].split(",")[b]), {}))
            per.append(summarize(bc.values(), rng, boot))
        blocks[lang] = per
        summary["cells"][f"B|{lang}|scale|blocks"] = [{k: v for k, v in s.items() if k != "reps"} for s in per]
        print(f"    {NAMES[lang]:11s} whole study {fmt(whole)}")
        print(f"    {'':11s} blocks " + ", ".join(f"{s['est']:.3f}" for s in per)
              + f" s -> {min(s['est'] for s in per):.3f}-{max(s['est'] for s in per):.3f} s")
    if "cs" in blocks and "py" in blocks:
        sps = [ratio(p, q) for p, q in zip(blocks["py"], blocks["cs"])]
        summary["derived"]["B|speedup|scale|blocks"] = sps
        print("    Optuna / OptunaSharp per block: " + ", ".join(fmt_ratio(r) for r in sps))

    # P - waiting trials.
    print("\n### P - Waiting trials (100 ms sleeps; ideal 1.0 s)")
    for jobs in (64, 128):
        cs = cell("P", "cs", "pool", f"jobs{jobs}", f"jobs{jobs}")
        py = cell("P", "py", "pool", "both", f"jobs{jobs}")
        ru = cell("P", "ru", "pool", "both", f"jobs{jobs}")
        for lang, s in (("cs", cs), ("py", py), ("ru", ru)):
            record(f"P|{lang}|jobs{jobs}", s)
        print(f"  {jobs * 10} trials, {jobs} workers:\n    OptunaSharp {fmt(cs)}\n    Optuna      {fmt(py)}\n    Rustuna     {fmt(ru)}")

    # C - large studies.
    print("\n### C - Large studies (40 floats, seed 0, defaults); fingerprints are trial hashes for OptunaSharp and Optuna")
    hashes = {}
    for cfg, label, trials in (("tpe", "TPE, 1,000 trials", 1000), ("motpe", "TPE, two objectives, 1,000 trials", 1000),
                               ("cmaes", "CMA-ES, 10,000 trials", 10000), ("nsga2", "NSGA-II, 10,000 trials", 10000),
                               ("cmaes100k", "CMA-ES, 100,000 trials", 100000),
                               ("nsga2100k", "NSGA-II, 100,000 trials", 100000)):
        base = cfg.replace("100k", "")
        cs = cell("C", "cs", "proof", cfg, f"{base}-plain-{trials}")
        py = cell("C", "py", "pbench", cfg, f"{base}-{trials}")
        ru = cell("C", "ru", "pbench", cfg, f"{base}-{trials}")
        for lang, s in (("cs", cs), ("py", py), ("ru", ru)):
            record(f"C|{lang}|{cfg}", s)
        sp = ratio(py, cs)
        summary["derived"][f"C|speedup|{cfg}"] = sp
        cs_h = set(cs["fingerprints"]) if cs else set()
        py_h = set(py["fingerprints"]) if py else set()
        hashes[cfg] = cs_h
        same = "n/a" if not py_h else ("EQUAL" if cs_h == py_h and len(cs_h) == 1 else f"differ cs={sorted(cs_h)} py={sorted(py_h)}")
        summary["derived"][f"C|hash|{cfg}"] = same
        print(f"  {label}:\n    OptunaSharp {fmt(cs)}\n    Optuna      {fmt(py)}\n    Rustuna     {fmt(ru)}")
        print(f"    Optuna / OptunaSharp {fmt_ratio(sp)} | OptunaSharp / Rustuna {fmt_ratio(ratio(cs, ru))} | hashes vs Optuna: {same}")
        c = cells.get(("C", "cs", "proof", cfg, f"{base}-plain-{trials}"))
        if c:
            for field, unit in (("retained", " MB"), ("peakWS", " MB")):
                fs = summarize(c.values(field), rng, boot)
                summary["cells"][f"C|cs|{cfg}|{field}"] = {k: v for k, v in fs.items() if k != "reps"}
                print(f"    OptunaSharp {field}: {fmt(fs, unit, 1)}")

    # D - compact 100k.
    print("\n### D - compactFinishedTrials, 100,000 trials (plain = block C)")
    for cfg, plain, base in (("cmaes100k-compact", "cmaes100k", "cmaes"), ("nsga2100k-compact", "nsga2100k", "nsga2")):
        c = cells.get(("D", "cs", "proof", cfg, f"{base}-compact-100000"))
        p = cells.get(("C", "cs", "proof", plain, f"{base}-plain-100000"))
        if not c or not p:
            continue
        sc, sp_ = c.stats(rng, boot), p.stats(rng, boot)
        record(f"D|cs|{cfg}", sc)
        same = "== plain" if sc["fingerprints"] == sp_["fingerprints"] else f"DIFFERS {sc['fingerprints']} vs {sp_['fingerprints']}"
        print(f"  {cfg}: time {fmt(sc)}\n    vs plain {fmt(sp_)}; hash {same}")
        for field, unit, digits in (("retained", " MB", 1), ("peakWS", " MB", 0), ("paused", " s", 2), ("opt_alloc", " MB", 0)):
            a, b = summarize(c.values(field), rng, boot), summarize(p.values(field), rng, boot)
            summary["cells"][f"D|cs|{cfg}|{field}"] = {k: v for k, v in a.items() if k != "reps"}
            summary["cells"][f"D|cs|{plain}|{field}"] = {k: v for k, v in b.items() if k != "reps"}
            print(f"    {field}: compact {fmt(a, unit, digits)} | plain {fmt(b, unit, digits)}")

    # E - executors.
    print("\n### E - Executors (DedicatedThreads = reused, ThreadPool = pool); rates = count / time")
    for m, count, unit in (("soak", 5000, "trials/s"), ("tpe", 2000, "trials/s"), ("manycalls", 200, "calls/s")):
        for ex in ("reused", "pool"):
            s = cell("E", "cs", "execprobe", f"{ex}-{m}", m)
            record(f"E|{ex}|{m}", s)
            if s and s["est"]:
                rate = {"est": count / s["est"], "lo": count / s["hi"], "hi": count / s["lo"]}
                summary["derived"][f"E|{ex}|{m}|rate"] = rate
                print(f"  {m} {ex}: {rate['est']:.0f} {unit} [{rate['lo']:.0f}-{rate['hi']:.0f}] from {fmt(s)}")
    for mult in ("2", "1"):
        for ex in ("reused", "pool"):
            c = cells.get(("E", "cs", "execprobe", f"{ex}-starve{mult}", "starve"))
            if not c:
                continue
            s = c.stats(rng, boot)
            record(f"E|{ex}|starve{mult}", s)
            nj = sorted({f.get("njobs") for runs in c.procs.values() for _, f in runs})
            mx = summarize(c.values("max"), rng, boot)
            p50 = summarize(c.values("p50"), rng, boot)
            summary["cells"][f"E|{ex}|starve{mult}|max"] = {k: v for k, v in mx.items() if k != "reps"}
            print(f"  starve x{mult} {ex} (nJobs {nj}): study {fmt(s)}; max delay {fmt(mx, ' ms', 1)}; p50 {fmt(p50, ' ms', 1)}")

    # F - per-worker.
    print("\n### F - Per-worker samplers")
    for (block, lang, probe, cfg, label), c in sorted(cells.items()):
        if block != "F":
            continue
        s = c.stats(rng, boot)
        record(f"F|{cfg}|{label}", s)
        print(f"  [{cfg}] {label.replace('_', ' ')}: {fmt(s)}")

    # G - compact table.
    print("\n### G - docs/performance.md compact table")
    for spec, trials in (("random", 10000), ("cmaes", 10000), ("tpe", 1000)):
        for mode in ("plain", "compact"):
            c = cells.get(("G", "cs", "proof", f"s12-{spec}-{mode}", f"{spec}-{mode}-{trials}"))
            if not c:
                continue
            s = c.stats(rng, boot)
            record(f"G|{spec}|{mode}", s)
            extra = []
            for field, unit, digits in (("retained", " MB", 1), ("peakWS", " MB", 0), ("opt_alloc", " MB", 0)):
                fs = summarize(c.values(field), rng, boot)
                summary["cells"][f"G|{spec}|{mode}|{field}"] = {k: v for k, v in fs.items() if k != "reps"}
                extra.append(f"{field} {fs['est']:.{digits}f}{unit}" if fs["est"] is not None else f"{field} -")
            print(f"  {spec} {mode}: time {fmt(s)}; " + "; ".join(extra) + f"; hash {s['fingerprints']}")

    # Validity: every flagged cell.
    print("\n### Validity")
    flagged = [(k, c.stats(rng, boot)["flags"]) for k, c in sorted(cells.items()) if c.stats(rng, boot)["flags"]]
    summary["flagged"] = [{"cell": "|".join(k), "flags": f} for k, f in flagged]
    if not flagged:
        print("  every cell passed: >= 5 processes, steady (within 3 %), no JIT (< 1 % of run time) or imports in measured "
              "runs, one fingerprint, 95 % CI half-width <= 5 %")
    for k, f in flagged:
        print(f"  {' '.join(k)}: {', '.join(f)}")

    # Interference: what the counted processes ran beside. Every one passed the guard (each window under the limits); this
    # shows how far under, and for the pinned blocks how busy the CPUs outside the pinned pair were (the runner's own shell
    # and the logger included, so a few % is the floor).
    if load_path:
        print(f"\n### Load beside the counted processes (load.log; {guard['interval']:g}-s windows overlapping each process)")
        worst = []
        for key in good:
            start, end, _ = spans[key]
            inside = overlapping(samples, start, end, guard["interval"])
            peaks = load_peaks(inside)
            foreign = None
            if key[0] in SERIAL_BLOCKS:
                foreign = statistics.fmean(statistics.fmean(c for i, c in enumerate(s[6]) if i not in (2, 3)) for s in inside)
            score = max(peaks[n][0] / guard[n] for n in ("guests", "services", "others") if peaks[n] is not None)
            worst.append((score, key, peaks, foreign))
        worst.sort(key=lambda w: w[0], reverse=True)
        summary["interference"] = [{"process": " ".join(k), "peaks": {n: (p[0] if p else None) for n, p in pk.items()},
                                    "foreign": f} for _, k, pk, f in worst[:20]]
        if worst:
            top = {n: max((pk[n][0] for _, _, pk, _ in worst if pk[n] is not None), default=None)
                   for n in ("guests", "services", "others")}
            print(f"  counted processes: {len(worst)}; highest window over all of them: "
                  + ", ".join(f"{n} {v:.1f}" for n, v in top.items() if v is not None) + " (% of one CPU)")
        for _, key, peaks, f in worst[:12]:
            print(f"  {' '.join(key)}: peak " + ", ".join(f"{n} {p[0]:.1f} at {p[1]}" for n, p in peaks.items() if p)
                  + (f"; other CPUs busy {f:.1f}% each on average" if f is not None else ""))

    if opts["--json"]:
        with open(opts["--json"], "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1, default=list)
        print(f"\nwrote {opts['--json']}")


if __name__ == "__main__":
    main()
