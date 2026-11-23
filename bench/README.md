# The README benchmark

The harness behind [README.md's Performance section](../README.md#performance): probes that time the same seeded workloads in
OptunaSharp (this checkout's `src/`), Python Optuna and Rustuna, a runner that schedules them on a quiet Linux machine, and a
parser that turns the output into the numbers, their confidence intervals and the checks that they are valid. Rustuna is
measured as a reference: the parser's report prints it, the README does not.

## What it measures

Steady-state speed: how fast each library runs a workload once its process is warm. Not start-up. Each process first
repeats the workload until it is warm and throws those runs away, then measures.

Until 2026-11-23 every workload was timed once, in a fresh process. On the worker machine about four fifths of
OptunaSharp's time in the short Samplers rows was then .NET's tiered JIT compiling OptunaSharp and NumSharp, and Optuna's
first runs included its lazy imports (torch for GP, cmaes for CMA-ES, SciPy's QMC engine). Those numbers compared start-up
costs, not the samplers. A short-lived process (one CLI command) does pay start-up; the parser still reports each process's
first run, as "first run", for that reason, but no table is built from it.

## Protocol

The same rules for every library, in `common/Bench.cs` (C#) and `py/benchlib.py` (Python), with the budgets the runner
exports:

| Step | Rule | Why |
|---|---|---|
| Warm-up | At least `BENCH_WARMUP_RUNS` runs (2) and `BENCH_WARMUP_SECONDS` (5 s); for .NET also at least `BENCH_DOTNET_TIERING_RUNS` runs (70); then on until the last quarter of the warm-up runs (`BENCH_WARMUP_QUIET_SHARE`, 0.25) compiled nothing (.NET) or imported nothing (Python); at most `BENCH_WARMUP_MAX_SECONDS` (60 s) once the first two minimums are met; printed, never counted | .NET's tiered compilation with Dynamic PGO moves a method to instrumented code after 30 calls and to tier 1 after 30 more, so a method called once per run settles only around run 61-63, with the JIT silent in between: OptunaSharp's 5,000-trial Random row compiles 96-176 methods per run at warm-up runs 31-36 and 12-58 at runs 61-68. A fixed 5-s warm-up (58 runs) measured that second step: runs up to 9 % slower. Python's first run imports lazily (torch for GP, cmaes for CMA-ES, SciPy's QMC) |
| Measure | At least `BENCH_MEASURE_RUNS` runs (3), then on until `BENCH_MEASURE_SECONDS` (10 s) or `BENCH_MAX_RUNS` (30) | Several runs per process make its median robust to one slow run and show whether the process was still warming up; a short workload fills its 10 s with up to 30 runs |
| Each run | A new sampler and study with the same seed; a full garbage collection before it, outside the timer; only `Optimize` / `optimize` is timed | Every run does the same work from the same heap state; a reused sampler would continue its random stream |
| Processes | `ROUNDS` fresh processes per workload (5), the rounds the outer loop | The variation between processes (JIT decisions, memory layout) belongs in the interval too |
| Order | The libraries take turns on each workload, the order reversed every round | A drift, a warm core or a busy hour favours nobody |
| Pinning | One-trial-at-a-time workloads on two cores whose hyperthread siblings stay idle (`taskset -c 2,3`); parallel ones on every CPU | The main thread has a core, the runtime's helpers (JIT, GC, OpenBLAS, Rust threads) the other, the same two for every library |
| Idle gate | Before every process: three 1-s windows under 5 % busy with the guests under half a CPU (fifteen after a voided attempt), or the run aborts after `GATE_MAX_WAIT` (2 h) | Numbers from a quiet machine or none, and every process starts from the same cooled-down state |
| Load guard | After every process: each 2-s window of `load.log` it overlapped must show the guests (containers and VMs), the host services and the other host sessions under half a CPU each (`GUARD_*`); otherwise the runner voids it and runs it again in its place, up to `ATTEMPTS` (6) attempts, and the parser drops any that stayed loaded | The gate sees only the moment before a process; on 2026-11-23 a TraderLab pass started a minute into block B and ran beside 13 processes for three minutes (guests at one CPU) |

The large studies warm up on the same study at a tenth of its trials (100 for the 1,000-trial studies): it runs the same
code, and Python Optuna's 1,000-trial studies take 38-197 s each, so full-size warm-up runs would triple the run. The
measured runs then show whether that was enough (the steady check below). The warm-up rules differ between the runtimes
only where the runtimes do: CPython 3.12 has no JIT and specializes its bytecode within a few executions, all inside its
first run, so its warm-up ends as soon as a run imports nothing; extra warm-up runs are never counted, so they cost time,
not fairness. `execprobe starve` measures how late other work
starts while a thread pool that has not grown yet is blocked; a second run in the same process would find a grown pool, so it
warms the JIT with a short non-blocking study and measures once per process, and the five processes are its samples.

## Statistics

`parse.py`, per cell (a library's workload):

- **Per process:** the median of its measured runs.
- **Estimate:** the trimmed mean of the processes' medians, the fastest and the slowest process dropped (with five processes,
  the mean of the middle three).
- **95 % confidence interval:** a hierarchical bootstrap: resample the processes with replacement, within each resample its
  measured runs, recompute the estimate; 4,000 seeded replicates, so the same results file always gives the same interval
  (Davison and Hinkley, *Bootstrap Methods and their Application*, 1997, section 3.8; the two-level design follows Kalibera
  and Jones, *Rigorous Benchmarking in Reasonable Time*, 2013).
- **Speedups:** Optuna's estimate over OptunaSharp's, with the interval of the ratio of bootstrap replicates (the two
  libraries ran in different processes, so independently). The nJobs speedups pair each process's 1-worker and n-worker
  medians.

Validity checks, each printed as a flag on the cell and listed at the end of the report:

| Flag | Fails when | It means |
|---|---|---|
| `drift` | The median over processes of each process's change across its measured runs, along their Theil-Sen trend line (the median pairwise slope), is more than 3 % | The process was still getting faster (not warm) or slower (memory growth); a workload that alternates between two times run by run (a collection cycle) does not trip it |
| `jit` | JIT compilation took over 1 % of OptunaSharp's measured time | A measured run was not warm |
| `imports` | A Python measured run imported a module | A measured run was not warm |
| `fingerprint` | Measured runs of a seeded workload printed different fingerprints (best value or trial hash) | The runs did different work |
| `ci` | The 95 % interval's half-width is over 5 % of the estimate | Too noisy to print to two digits |
| `processes` | Fewer than five processes counted | Not enough processes for the trimmed mean |

Processes are dropped, and listed, when they exit with an error, ran on other CPUs than the runner meant (each prints its
own affinity), or, with `load.log`, fail the load guard or have no load window at all (the logger was not running). The
runner's voided attempts are listed too: their output stays in `results.txt` but is never counted. A parallel-trials
process counts only if its 1-worker time is within 10 % of 64 × 50 ms (its loop calibration held). Each cell also shows
how many warm-up runs its processes made and why each warm-up ended (`quiet`, or a cap), and the report ends with the
highest guest, service and other-session load beside any counted process.

## Where each README number comes from

| README | Block | Probes |
|---|---|---|
| Parallel trials | A | `probes/njobs`, `py/njobs_scaling.py` |
| Parallel trials, trials that wait | P | `probes/pool`, `py/pool.py` |
| Samplers | B | `probes/perf`, `probes/wide`, `probes/scale`; `py/perf.py`, `py/wide.py`, `py/scale.py` |
| Large studies | C | `probes/proof`, `py/pbench.py` (also the seeded-trial hashes) |
| Memory paragraph | D | `probes/proof` with `compactFinishedTrials` |
| Executors | E | `probes/execprobe` |
| Per-worker samplers | F | `probes/modeprobe` |
| docs/website-src/docs/performance.md, compact storage | G | `probes/proof` |

## Running it

The worker machine (one Intel Xeon E5-2690 v2, 10 cores / 20 threads; its rules are in the homelab's proxmox repository) is
where the README's numbers come from; any Linux machine with a quiet second core works.

1. `bash bench/publish.sh` builds every probe for linux-x64 (Release, framework-dependent) and writes
   `bench/harness-stage.tar`, with a `SOURCE` file naming the commit of `src/` it was built from.
2. Copy that tar, `bench/setup.sh`, the .NET 10 runtime, a CPython 3.12 build and the pinned wheels (optuna 5.0.0,
   rustuna 0.3.1, numpy 2.5.4, scipy 1.16.3, cmaes 0.13.1, torch 2.12.1+cpu, psutil) into `$BENCH/incoming`, then run
   `bash setup.sh`. It builds the runtime and the venv once and unpacks the harness into `$BENCH/v2`.
3. Start the run detached: `RUN_NAME=<name> nohup setsid bash $BENCH/v2/run_worker.sh </dev/null >run.out 2>&1 &`.
   `BLOCKS` (default `ABPCDEFG`), `ROUNDS`, the `BENCH_*` budgets and the gate's and guard's limits are environment
   variables. Progress is in `results/<name>/run.log` (voided attempts and long gate waits included), which ends with
   `ALL DONE` or `ABORTED`. To stop a run, signal the runner's process group: `kill -- -<pgid of bash v2/run_worker.sh>`.
   A run that aborted or was stopped continues with `RESUME=1` and the same `RUN_NAME`: every process with a clean verdict
   is kept, and every other one runs again in its place.
   While it runs, keep the host itself quiet too: the guard counts other host sessions, so reading `run.log` with `tail`
   over ssh is fine (a few % of a CPU for a moment), but one call of Proxmox's Perl tools (`pct exec`, `pct list`, `qm`)
   takes about a CPU-second and voids the process it lands beside (seen on 2026-11-23: 57 % of a CPU in one window).
4. `python -I $BENCH/v2/parse.py results/<name>/results.txt results/<name>/load.log --json summary.json` prints the report
   and writes every cell to JSON.

## Files

| File | What it is |
|---|---|
| `common/Bench.cs`, `py/benchlib.py` | The protocol: warm-up, measured runs, one line per run |
| `common/ProbeEnv.cs`, `py/pyrun.py` | Print each process's runtime and CPU affinity first (and opt a Windows process out of power throttling) |
| `probes/*/` | The C# probes, each a project against `../../../src` (not in the solution) |
| `py/*.py` | The Python probes, for Optuna and Rustuna |
| `run_worker.sh` | The runner: blocks, rounds, pinning, gates, the load log |
| `monitor.py` | The idle gate and the load log |
| `parse.py` | The statistics, the checks and the JSON |
| `publish.sh`, `setup.sh` | Build the harness; set it up on the machine |
