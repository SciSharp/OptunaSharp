# OptunaSharp — Threading Design

**Status:** describes the code at `f39763d` (2026-10-30), plus two changes of 2026-11-06: the `nJobs` executor,
which reuses its worker threads (review item P3-4, §3.5–3.6), and the GIL-aware monitor entry, which lets code
that holds the Python GIL call OptunaSharp (review item P3-1, §8). Three items are open. Each is marked where
it applies, and §12 lists them all:

- `PymooSampler` driven from more than one thread (review item 19b, §8);
- the 32 bits of entropy behind the samplers' seeds from the operating system (§6.3);
- the terminator callback's unguarded random generators (§6.6).

Optuna gets most of its thread safety from Python itself. Only one thread runs Python code at a time
(the GIL), and numpy's `RandomState` locks every draw. .NET threads run in parallel, so OptunaSharp adds
explicit monitors wherever Optuna relied on those two. This document describes how the work of a study
is spread over threads, which locks exist and in which order they are taken, what each layer
guarantees, and the rules for code that runs inside OptunaSharp: objectives, callbacks, samplers,
pruners, storages and log handlers.

The user-facing summary is the site's
[Concurrency section](website-src/docs/storage.md#concurrency-workers-studies-and-languages-on-one-journal)
on the Storage page. This file is the design behind it.

---

## 1. Principles

1. **Parallelism is per trial.** `Optimize(nJobs: …)`, several processes and several machines run
   trials side by side. Inside a trial, OptunaSharp's own numeric code is single-threaded (the NumSharp
   house rule; §9).
2. **Optuna's behavior is the specification.** The `nJobs` loop, its error and stop rules, the journal
   protocol and the per-thread worker identity are ports of Optuna's code. Where parallel .NET threads
   expose a race that the GIL hides in Python, OptunaSharp adds a lock and keeps the outcome that Python
   shows in practice.
3. **A lock costs nothing with one thread.** Every monitor is uncontended in a sequential study, so a
   seeded `nJobs: 1` run stays bit-identical to Optuna.
4. **Everything is synchronous.** The objective (`Func<Trial, double>`), the samplers, the pruners and
   every `BaseStorage` are synchronous, as in Optuna. A running trial occupies a thread for its whole
   life. The only asynchronous API is `DisposeAsync` on the storages that own a connection.
5. **A sampler or pruner instance is serialized, not partitioned.** Its monitor admits one call at a
   time from every thread and every study that uses the instance.
   - A stateless sampler can therefore serve two studies at once. One `RandomSampler` shared by two
     studies on two threads draws 400 distinct values in 2 × 200 trials, as in Optuna.
   - A stateful sampler serves a second study only if it keys its state by study. Samplers built on
     `PerStudyState<T>` do (§6.3).
   - Samplers whose relative search space is an `IntersectionSearchSpace`, or a variant of it, raise
     Optuna's ``ValueError: `IntersectionSearchSpace` cannot handle multiple studies.`` That check
     compares study ids. Study ids repeat across in-memory storages, so there the check cannot fire,
     and the sampler works from a search space mixed from both studies, as in Python.

   Give each study its own instance.

## 2. The threads

| Thread | Started by | Lives for | Runs |
|---|---|---|---|
| The caller's | your code | — | `Optimize`. With `nJobs: 1` it runs every trial, callback and progress-bar update itself; otherwise it runs the submit loop and waits. `Ask` and `Tell` run on whichever thread calls them. |
| `nJobs` worker | `OptimizeLoop.StartWorker`: a `LongRunning` task, which the default scheduler runs on a new background thread named `.NET Long Running Task` | one trial | the reseed, `Ask`, the objective, `Tell`, the callbacks, the progress bar |
| Heartbeat | `HeartbeatThread.Start`, for each trial of an RDB storage whose heartbeat interval is set (§6.7) | the trial's objective | the trial's heartbeat, once at the start and then once per interval. A background thread named `OptunaSharp heartbeat (trial_id=N)`. |
| Storage client I/O | StackExchange.Redis, NRedisStack, Grpc.Net.Client, the AWS SDK | the connection | socket reads and writes, on the .NET thread pool (§7.2) |
| gRPC proxy server | Kestrel | one request | the wrapped storage's operation, synchronously, on a pool thread |
| OpenBLAS | the native library | the process | LAPACK and BLAS internals (§9) |
| Python | embedded CPython | the process | every bridge call, under the GIL (§8) |

OptunaSharp starts no other threads: no timers, no `Task.Run`, no thread-pool work items. Logging and the
progress bar write on the thread that logs.

**What a worker inherits.** A worker starts with the caller's `ExecutionContext`, as every .NET task does:
its `CurrentCulture` and `CurrentUICulture`, and every `AsyncLocal<T>` value (which is also how
`Activity.Current` and logging scopes travel). Measured: the workers and callbacks of a caller running
`de-DE` ran `de-DE` and saw the caller's `AsyncLocal` value. An objective therefore formats and parses
numbers the same way under `nJobs` as sequentially. Python's executor threads start without the caller's
`contextvars`, and Python's locale is process-wide.

**Background threads.** Workers and heartbeat threads never keep a process alive. `Optimize` waits for
its workers before it returns (§3.1), so this matters only when the process ends while `Optimize` runs
(§3.4).

## 3. `Optimize(nJobs)`

[`Internal/Optimize.cs`](../src/OptunaSharp.Core/Internal/Optimize.cs) is a port of
`optuna/study/_optimize.py`.

### 3.1 The loop

- `nJobs: 1`, the default, runs `OptimizeSequential` on the caller's thread.
- `nJobs: -1` uses `Environment.ProcessorCount` workers (Python: `os.cpu_count()`).
- `nJobs` of 0 or below -1 raises `ValueError: max_workers must be greater than 0`, the refusal of
  Python's `ThreadPoolExecutor`.
- Any other value runs Python's submit loop on the caller's thread:

```text
for each submission:
    stop when Stop() was called, the timeout has passed, or nTrials trials were submitted
    when nJobs trials are in flight:
        wait for the first to finish (Task.WhenAny) and raise the first failure among the finished ones
    start a worker for one trial
finally: wait for every worker still running, without raising (the exit of Python's `with ThreadPoolExecutor`)
```

At most `nJobs` trials are in flight at once. The loop starts one worker per submission, so an `nJobs`
above `nTrials` runs `nTrials` trials on as many threads, and `nTrials: 0` runs none, as Python's executor
starts its threads only on demand.

### 3.2 What a worker does

A worker is Python's `_optimize_sequential(n_trials=1, reseed_sampler_rng=True)`, step for step:

1. It sets its thread's `InOptimizeLoop` flag (§4) and puts the previous value back when it ends
   (`restoreLoopFlag`).
2. It reseeds the study's one sampler (`ReseedRng`) under the sampler's monitor.
   - As in Optuna, the reseed takes fresh entropy from the operating system, which is why a seeded
     sampler's parallel run cannot be reproduced.
   - The reseed does not keep concurrent trials apart. The workers reseed and draw from the one shared
     generator in turn, and the monitor is what makes their draws distinct (§6.3).
   - The new seed has 32 bits (open, §6.3).
3. It runs one trial (`RunTrial`, Python's `_run_trial`):
   - with heartbeats on, it first fails the stale trials (§6.7);
   - it asks for a trial, starts the heartbeat thread and runs the objective;
   - it joins the heartbeat thread, tells the result under the trial's tell stripe (§6.5) and logs the
     outcome;
   - it rethrows an objective exception that `catch` does not cover (§3.3).
4. It finishes the trial in Python's order:
   - `GC.Collect()` when `gcAfterTrial` is set, in a `finally`, so after a failed trial too;
   - the callbacks in list order, each with its own deep copy of the stored trial;
   - the progress bar.
5. When its loop ends without an exception, it calls the storage's `RemoveSession`, as Python does.
   Every OptunaSharp storage inherits the empty implementation from `BaseStorage`. Python's
   `RDBStorage` drops the thread's SQLAlchemy session there, and the C# RDB storages hold no per-thread
   session: one connection per SQLite storage, a `DbContext` per EF Core operation.

### 3.3 Errors

What happens to the trial, and whether `Optimize` raises. Every row was observed on Optuna 5.0 too, and
Optuna does the same except where a row says otherwise. `OptimizeEdgeCaseTests` and
`OptimizeConcurrencyTests` pin every row but the last.

| What fails | The trial | Raised by `Optimize` |
|---|---|---|
| the objective, with an exception type in `catch` | `FAIL`, logged | no |
| the objective, with any other exception (a nested `Optimize` inside a worker's objective fails this way, with ``RuntimeError: Nested invocation of `Study.optimize` method isn't allowed.``) | `FAIL`, logged | yes |
| the objective, with `TrialPruned` | `PRUNED` | no |
| the objective's result: `NaN`, the wrong number of values, or `null` (±∞ are values) | `FAIL`, logged with the reason | no |
| the sampler, while the objective suggests | as the objective failing: `catch` applies, and the sampler's monitor is released | per `catch` |
| the sampler's `BeforeTrial`, inside `Ask` after the trial was created | stays `RUNNING` | yes, whatever `catch` holds |
| the sampler's `AfterTrial`, inside the tell | stored, because the state is written in a `finally` | yes, whatever `catch` holds |
| a callback | stays `COMPLETE`, because callbacks run after the trial is stored | yes, whatever `catch` holds |
| the storage, writing the final state | stays `RUNNING` | yes: the storage's own exception, which `catch` never covers. A documented divergence (`[Misaligned]`): Optuna 5.0 raises `AssertionError('Should not reach.')` instead. The logging `finally` of `_run_trial` finds no objective error and no warning to log, and hits `assert False`. |
| the tell, because `FailStaleTrials` failed the trial while its objective ran (§6.7) | `FAIL` | yes: `ValueError: Cannot tell a FAIL trial.` The same divergence: Optuna 5.0 raises `AssertionError('Should not reach.')` by the same route (measured 2026-10-30, sequentially and at `n_jobs=2`). |

How a raised exception travels under `nJobs`:

- A worker rethrows the exception. The submit loop raises it the next time it waits, after the trials
  still running have finished. It raises the original exception, not an `AggregateException`.
- When several trials failed by the time of one wait, the first is raised. The others are observed and
  dropped, so none resurfaces through `TaskScheduler.UnobservedTaskException`.
- Failures among the trials that finish after the last submission are not raised, because Python never
  calls `result()` on them. They are still stored as `FAIL` and logged.
- With `nJobs: 1` the first raised exception ends the run at once.

### 3.4 Stopping

- `Stop()`, the timeout and `nTrials` stop the *scheduling*, and the trials in flight finish.
- `nTrials` caps the submissions, so an `nJobs` run never exceeds it. A worker that starts after `Stop()`
  or after the deadline runs no trial, so a stopped or timed-out run can end below it.
- A `Stop()` from a callback, such as `MaxTrialsCallback`'s, comes only when its trial ends, so an `nJobs`
  run can end up to `nJobs - 1` trials past that limit, as in Python.
- The timeout is measured from one start time that the submit loop and the workers share. Both read
  `DateTime.Now`, the local wall clock, as Python reads `datetime.now()`. A clock step or a
  daylight-saving change during a run therefore moves the deadline.
- `Stop()` works only on a thread inside the optimize loop, that is from an objective or a callback.
  Called from any other thread it raises Python's
  ``RuntimeError: `Study.stop` is supposed to be invoked inside an objective function or a callback.``,
  and the run goes on.
- The stop flag is a `volatile` field of the `Study` object. It stops every `Optimize` call running on
  that object, and each new `Optimize` call clears it, as Python resets `_stop_flag`.
- There is no `CancellationToken`. To cancel from outside, set a flag that a callback checks, and let the
  callback call `Stop()`.
- **Ctrl+C.** .NET ends the process on Ctrl+C by default, and the worker threads end with it. The trials
  in flight stay `RUNNING` in a persistent storage: an RDB storage with heartbeats fails them later as
  stale, and a journal never does.
  - Python, when `n_jobs` is 1, stores the running trial as `FAIL` when the interrupt arrives in its
    objective. With more workers it lets the running trials finish. Either way it then raises
    `KeyboardInterrupt`.
  - To let the trials in flight finish instead, handle `Console.CancelKeyPress`: set `e.Cancel = true`
    and the flag your callback checks.

### 3.5 Why workers run on dedicated threads

Until `3a75811` the workers ran on the .NET thread pool (`Task.Run`). That caused two failures:

- **Pool starvation.** A worker is blocked for most of a trial: on the storage's lock, on file or
  network I/O, in the objective. With `nJobs` above the pool's minimum thread count every pool thread was
  blocked, and the clients that complete their synchronous calls on the pool stalled (§7.2).
  `Optimize(nJobs: 96)` on a StackExchange.Redis journal died after 88 trials with
  `RedisTimeoutException`.
- **Stale thread-local state (review item H5).** A pool thread outlives the call, so a worker's
  `InOptimizeLoop = true` stayed on the thread: 61 of 64 later `Optimize` calls on pool threads failed
  with a false "Nested invocation" error. Workers now restore the flag (`restoreLoopFlag`), whatever
  thread runs them.

`3a75811` then gave every trial a new thread (`TaskCreationOptions.LongRunning`). Since 2026-11-06,
`StartWorker` queues each worker on `TrialWorkerScheduler.Shared` instead: OptunaSharp's own dedicated
threads, never pool threads. They are created on demand, reused by later trials and later `Optimize` calls,
and retired after 30 s idle (background threads, so they never keep the process alive).
- **No inlining:** a trial is never run on the thread that waits for it.
- **No stale state:** each worker restores `InOptimizeLoop` and drops its `CachedAllTrials` snapshot when it
  finishes.
- **Pinned by tests:** `NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool`, and `TrialWorkerSchedulerTests`
  (reuse, `nJobs` really in flight, no inlining, retirement).

### 3.6 The executor's throughput (P3-4, fixed 2026-11-06)

Python's `ThreadPoolExecutor` creates up to `n_jobs` threads and reuses them for the whole call. Until
2026-11-06 that reuse was not ported. Creating a thread costs about 100 µs, serialized on the submit loop,
which capped `nJobs` near 10,000 trials/s whatever the trials did. Cheap trials at `nJobs: 8` were slower
than sequential ones (Random sampler, in memory: 4.4k–6.2k trials/s against 22k–26k at `nJobs: 1`).
Trials that take a millisecond or more did not notice it.

The fix is the "OptunaSharp threads, reused across calls" column below. Measured on 2026-11-06, the same
benchmark against the previous commit and the fix, in the same session:

| | Thread per trial (before) | Reused threads (now) |
|---|---|---|
| 5,000 cheap Random trials, `nJobs: 8` | 3.1k–5.0k trials/s | 8.2k–12.1k trials/s |
| 200 × `Optimize(nTrials: 8, nJobs: 8)` | 338–392 calls/s | 1,180–1,875 calls/s |

A replacement executor had to keep five properties. Each one is a measured failure of an alternative:

1. **Python's loop semantics** (§3.1–3.4).
2. **Clean per-thread state for every trial** (§4). A reused thread must leave `InOptimizeLoop` as it
   found it.
3. **`nJobs` trials really in flight.** Python runs `n_jobs` threads, and an `n_jobs` above the core
   count is common for objectives that wait on remote work.
4. **Never more parked pool threads than the pool has.** StackExchange.Redis, Grpc.Net.Client and the
   AWS SDK complete synchronous calls on the pool (§7.2).
5. **The pool stays available to the host.** A web application's requests and timers must not queue
   behind trials.

The alternatives were measured on 2026-10-29 with scratch builds of the real `Optimize`, on 32 logical
CPUs (the pool's minimum is 32), on a machine busy with other work:

| | Thread per trial (current) | Thread pool | Pool + StackExchange.Redis `DedicatedThreads` | Pool, minimum raised by `nJobs` | OptunaSharp threads, reused across calls |
|---|---|---|---|---|---|
| Redis journal, 400 trials, `nJobs` 64 / 96 | all complete | **58/400 and 85/400, `RedisTimeoutException`** | all complete | all complete | all complete |
| Trials in flight at `nJobs: 96` (960 × 1 s, in memory; ideal 10 s) | — | **at most 34; 29.4 s** | the same as the pool | 96; 10.4 s | 96; 10.3 s |
| gRPC proxy, 400 trials, `nJobs: 96` | — | **54.5 s** | the same as the pool (the flag is Redis-only) | 0.6 s | 0.6 s |
| Delay before the host's other pool work starts, `nJobs` 64–96 | — | up to 58 s | **0.5–2.9 s** | at most 85 ms | at most 16 ms |
| Process-wide setting required | none | none | a StackExchange.Redis feature flag | `ThreadPool.SetMinThreads` | none |

A dash means the configuration was not part of that run. The current thread per trial runs all `nJobs`
trials at once (128 trials of 1.5 s at `nJobs: 64` took 3.2 s, ideal 3.0 s) and kept the host's work
waiting at most 3 ms.

Throughput for cheap trials (in memory, `nJobs: 8`): the pool and reused threads tie at 7.8k–8.7k and
7.9k–8.6k trials/s, against 1.9k–6.0k for a thread per trial. For 200 calls of
`Optimize(nTrials: 8, nJobs: 8)`: threads reused across calls 813 calls/s, the pool 789, a thread per
trial 443, threads created per call 81. Threads reused across calls were chosen (2026-11-06): they are the
only option that keeps all five properties and matches the pool's throughput. The pool fails properties 4
and 5, so it can only ever be an opt-in; whether to expose one, and how, is the owner's open decision.

## 4. Per-thread state

**`Study.ThreadLocal`** is a `ThreadLocal<ThreadLocalStudyAttribute>`, one per `Study` object, the port
of Python's `study._thread_local`. It holds two fields:

- **`InOptimizeLoop`** is true while this thread runs an objective or a callback of this study.
  - It refuses a nested `Optimize` on the same thread (``Nested invocation of `Study.optimize` method
    isn't allowed.``), it allows `Stop()`, and it silences `Ask`'s heartbeat warning.
  - Because it is per thread, two threads may run `Optimize` on one study at the same time.
  - It is also per `Study` object, as in Python: a second `Study` object loaded for the same study has
    flags of its own.
- **`CachedAllTrials`** is one coherent snapshot of the study's trials. A sampler may read it many times
  while it suggests one trial. `Ask` and every tell clear it, so every trial starts from fresh history.

**The journal's worker id.** Every journal record carries `worker_id`, which is
`<storage guid>-<managed thread id>` (Python: `threading.get_ident()`). The replay credits a record to
"this worker" only when both parts match. The threads of one process therefore compete for an enqueued
trial exactly as separate processes do: the `RUNNING` record that lands first in the journal owns the
trial, and the others skip it.

## 5. Locks

### 5.1 Inventory

| Lock | Where | Protects | Held during I/O |
|---|---|---|---|
| Tell stripe | `BaseStorage.TellLockFor(trialId)`: 64 monitors per storage instance, picked by trial id | a tell's "still running" check, the sampler's `AfterTrial` and the final state write | yes, the storage writes |
| Trial monitor | `Trial._syncRoot`, one per `Trial` | the trial's caches; every check-then-write (report, suggest, user attribute, constraint) | yes, storage calls inside |
| Sampler monitor | `BaseSampler.SyncRoot`, one per sampler instance | every call the study makes into the sampler, and its random generator | yes, samplers read the study |
| Pruner monitor | `BasePruner.SyncRoot`, one per pruner instance | `Prune` and `FilterStudy` | yes |
| Sampler-private locks | `BaseGASampler`, `JanusSampler`, `JanusDeSampler`, `MoeaDSampler`, `SmsEmoaSampler`, `RandomEmbeddingSampler`, `OptunaHubSampler` | per-study caches | taken inside the sampler monitor; `OptunaHubSampler`'s inside the GIL too |
| The GIL | CPython | Python objects | taken by the bridges inside the sampler or pruner monitor; released by a thread that holds it while it waits for any lock above (§8) |
| Storage monitors | `InMemoryStorage`, `JournalStorage`, `CachedStorage`, the SQLite `RDBStorage`, the gRPC client's trial cache | the storage's in-memory state | all but `InMemoryStorage`'s: the journal's spans every round trip of an operation, SQLite's every database call, and `CachedStorage`'s and the gRPC cache's the read of new and unfinished trials |
| Journal file locks | `JournalFileBackend`'s offset-cache lock (short), and the lock file (`JournalFileSymlinkLock` or `JournalFileOpenLock`) around each append | the offset cache; one append at a time across processes | the lock file spans the write and its `fsync` |
| Database locks | EF Core: `SELECT … FOR UPDATE` row locks and a PostgreSQL advisory lock for schema creation; SQLite's database lock | arbitration between storage instances and processes | yes |
| Progress bar | `ProgressBar._updateLock` | the count and the line | reads the best trial |
| Logging configuration | `OptunaLogging._lock` | the level and the handler list | no: handlers run outside it |
| Python start-up | `PythonRuntime._gate` | starting the engine | once |

Every monitor is a .NET monitor, so a thread can re-enter a monitor it already holds. The four that come
before the GIL in the lock order (§5.2) are entered through `MonitorGate` instead of a C# `lock`: the tell
stripes, the trial monitor, and the sampler and pruner monitors. Without a Python bridge that is a plain
`Monitor.Enter`; with one, a thread that holds the GIL releases it while it waits (§8). The others stay
plain `lock`s, because no holder of theirs ever waits for the GIL.

**Shared state without a lock.** A few fields are shared across threads by design, each safe on its own
terms:

- `Study.StopFlag`, a `volatile` bool (§3.4).
- Caches built on first use and published whole through one reference, so a racing reader sees the
  finished object or builds an identical one:
  - `CategoricalDistribution`'s index of its choices, and `HyperbandPruner`'s bracket plan (both
    `volatile`);
  - the static QMC tables: the Sobol' direction numbers and the Halton primes.
- `ConcurrentDictionary` caches:
  - `EnumCategorical`'s value names (static);
  - `RandomEmbeddingProjection`'s rows;
  - `PythonRuntime`'s imported modules (static);
  - the EF Core storage's PostgreSQL data sources (static, each behind a `Lazy`, so concurrent first
    opens of one database build one).
- One-way switches set with `Interlocked`: the default journal lock's switch to the open-file protocol
  (§7.1), and the Redis backends' disposed flags.
- The heartbeat thread's stop signal, a `ManualResetEventSlim`.

The internal concurrent collections in `OptunaSharp.Collections` (a vendored `ConcurrentDictionary` and
`ConcurrentOrderedDict`) are not used by the library yet; `ConcurrentOrderedDictTests` pins their
contract.

### 5.2 Lock order

Outermost first. A thread that holds a lock takes only locks further down the list:

1. the tell stripe;
2. the trial monitor;
3. the sampler monitor or the pruner monitor, never one inside the other;
4. inside a sampler: its private locks, and for the Python bridges the GIL. `OptunaHubSampler` goes the
   furthest: the monitor, then the GIL, then its private `_gate`, then the storage, which it reads to
   mirror the study into its Python shadow study;
5. the storage monitor: `CachedStorage`'s before its backend's, `JournalStorage`'s before its backend's
   offset-cache lock and lock file;
6. the database and cross-process locks.

The progress bar takes its lock before it reads the study. Log handlers are called outside the logging
lock, but on the thread that logs, which may hold any of the locks above: a journal storage logs a new
study from inside its monitor.

### 5.3 Rules that follow from the order

- **A sampler or pruner hook must not tell a trial of the study it serves.** `Tell` takes a tell
  stripe, which comes before the sampler monitor, so a hook that tells can deadlock against a worker
  that is telling. A wrapper that drives private shadow studies may tell those: their storages and
  stripes are its own.
- **Code that holds the GIL never blocks on a lock that comes before it** (§8). OptunaSharp keeps this
  rule itself: the tell stripes, the trial monitor and the sampler and pruner monitors are entered through
  `MonitorGate`, and `Optimize(nJobs > 1)` waits for its workers through it too. Once a Python bridge has
  started (`PythonRuntime.Initialize`), a thread that holds the GIL releases it while it waits. Suggest,
  report, should-prune, ask, tell and optimize are therefore safe inside `using (Py.GIL())`. A lock of your
  own that a GIL-taking sampler or pruner holds while it takes the GIL must never be waited for by code that
  holds the GIL.
- **A sampler that wraps another calls it directly.** The outer monitor already covers the inner
  sampler.
- **Log handlers must not throw, call OptunaSharp, or take the GIL.** They can run under any lock
  above, and warnings reach them the same way. `Emit` does not catch, so a handler's exception
  surfaces in the operation that logged: a storage call, a tell, the optimize loop.

## 6. What each layer guarantees

### 6.1 Study

- Any number of threads may use one `Study` object: `Optimize` (two threads at once is fine, and the
  trials are numbered contiguously), `Ask`/`Tell`, `EnqueueTrial`, and reading `Trials` or `BestTrial`.
- A reader that polls the study during an `nJobs` run always sees a consistent study. The trials are
  numbered contiguously, every `COMPLETE` trial has its value and completion time, and the best value
  never gets worse than a completed trial the reader saw before.
- `Sampler` and `Pruner` are settable. Each call reads the property once, so the monitor it takes
  belongs to the instance that runs.
- Every enqueued trial is popped exactly once. `PopWaitingTrialId` claims a `WAITING` trial with
  `SetTrialStateValues(RUNNING)`, which the storage arbitrates. A worker that loses the race gets `false`
  or an `UpdateFinishedTrialError`, and tries the next trial.
- `EnqueueTrial(skipIfExists: true)` checks the existing trials and then enqueues, in two steps, as in
  Python. Two threads enqueueing the same parameters at the same moment can therefore both enqueue them.
- `InMemoryStorage` replaces a trial's record on every update instead of changing it (copy-on-write), so
  a record read with `deepcopy: false` never changes under its reader. Records read that way are the
  storage's own, shared by every thread: never mutate them (§10).

### 6.2 Trial

- A `Trial` serializes its own operations: suggest, report, should-prune, user attributes, constraints
  and the relative sample. Several threads may drive one trial:
  - the first report of a step wins, as Optuna documents (Python's own `Trial` races there: in 63% of
    contended steps on Optuna 5.0 the last report won);
  - one name suggested from several threads is sampled once;
  - the relative search space is inferred and sampled in one sampler call.
- A trial caches a detached copy of its stored record, so its writes never change a snapshot that
  another thread took before them (TPE's constant liar reads such snapshots).
- `Params`, `Distributions` and `UserAttrs` return copies.

### 6.3 Samplers

- The sampler monitor covers `BeforeTrial`, `InferRelativeSearchSpace` together with `SampleRelative`,
  `SampleIndependent`, `AfterTrial` and `ReseedRng`. `Trial`, `Study.Tell` and the `nJobs` workers take
  it.
- Why: NumSharp's MT19937 takes no lock, where numpy's does.
  - Two threads drawing at once produced identical parameters (1–7 duplicates per 400 trials at
    `nJobs: 16`), and the samplers' caches raced.
  - One unguarded generator drawn by 8 threads at once (200,000 draws each) repeated 30,000–40,000
    values and threw 1–11 `IndexOutOfRangeException`s per run (measured 2026-10-30).
- Consequence: the sampling of concurrent trials is serial, while their objectives run in parallel. At
  a high `nJobs`, a slow model-based sampler (GP, SAASBO, TPE over a long history) becomes the bottleneck.
- Per-study state lives in `PerStudyState<T>`, keyed by the storage object (held weakly) and the study
  id, because study ids repeat across storages. `TrialAnchor` detects a deleted study whose id was
  reused. The class is not synchronized itself: the sampler monitor covers it.
- Random numbers come from `LazyRandomState`, MT19937, never `System.Random`. An unseeded sampler seeds
  itself from OS entropy (`RandomNumberGenerator`) rather than the clock. Samplers created at the same
  moment therefore draw different streams: in one process, in worker processes, and in parallel
  `optunasharp ask` calls. On the clock, 8 parallel shells had repeated 11 and 15 of their 80 draws.
- **Open: 32 bits of entropy.** The seeds that `LazyRandomState` and every `ReseedRng` take from the
  operating system have 4 bytes: the seed of an unseeded sampler, and the new seed of each reseed.
  numpy's `RandomState()` and `RandomState.seed()` instead fill the generator's whole 624-word key from a
  128-bit `SeedSequence`.
  - Two of N 32-bit seeds are equal with probability about N²/2³³: 1% at 9,300 seeds, 50% at 77,000.
  - A generator reseeded with a seed it had before replays the earlier stream. A trial that draws first
    after each of the two reseeds draws what the earlier trial drew: a `RandomSampler` point, or a
    sampler's random-startup or independent values.
  - Measured 2026-10-30: one million `RandomSampler.ReseedRng` calls, each followed by one draw,
    repeated 116 draws, where the birthday bound predicts 116.4. One million numpy reseeds repeated none.
  - An `nJobs` run reseeds once per trial, so a run of 100,000 random trials can repeat a point. Across
    processes the risk is small: 1,000 unseeded workers share a stream with probability about 0.01%.
  - The fix is to seed from at least 128 bits, through `RandomState.seed(uint[])` (numpy's
    `init_by_array`).
- Some samplers can give one point to two concurrent trials, by design: JANUS without `nJobs` lanes,
  REMBO, JANUS-DE, LM-CMA, TuRBO, the genetic samplers, Grid, BruteForce and QMC. The Storage page lists
  them with counts.

### 6.4 Pruners

- The pruner monitor covers `Prune`, called from `ShouldPrune`, and `FilterStudy`, called from the trial
  and tell paths. Without it, `HyperbandPruner` built its brackets from several threads at once, which
  crashed `Optimize` at `nJobs` above 1.
- Pruning decisions of concurrent trials are therefore serial per pruner.

### 6.5 Tell

- A tell runs under its trial's stripe: one of 64 monitors per storage instance, picked by trial id. The
  "still running" check, the sampler's `AfterTrial` and the final state write happen together.
- Of several tells of one trial, from any thread and from the optimize loop, exactly one stores its
  result and reaches `AfterTrial`. The others are refused with `ValueError: Cannot tell a COMPLETE trial.`,
  or return the stored trial when `skipIfFinished` is set. Before 2026-10-28, about one round in 150
  passed both tells.
- The scope is one storage instance. Tells through two instances or two processes are arbitrated by the
  storage, as in Python, and the loser gets `UpdateFinishedTrialError`.
- RDB storages are wrapped in a new `CachedStorage` for each `Study` object (`StorageFunctions.GetStorage`,
  like Python's `get_storage`). Two `Study` objects on one RDB storage are therefore two instances here.

### 6.6 Callbacks, logging, warnings and the progress bar

- **Callbacks** run on the worker thread that ran the trial, after the trial is stored, each with its
  own deep copy of it, as in Python. Under `nJobs` the callbacks of different trials run at the same
  time, so a callback must be thread-safe.
- **Open: `TerminatorCallback` is not thread-safe.**
  - Its terminator's improvement evaluators (`RegretBoundEvaluator`, `EMMREvaluator`) each keep one
    `LazyRandomState` across calls, and `Invoke` takes no lock. At `nJobs` above 1, concurrent callbacks
    therefore draw from one unguarded generator (§6.3).
  - Measured at `nJobs: 8`: eight callbacks ran at once, and 8 rounds showed no failure, likely because
    the draws are a small part of each GP fit. In Python, numpy's lock makes the same draws safe.
  - Until the callback takes a lock, serialize it yourself:

    ```csharp
    var terminatorCallback = new TerminatorCallback();
    var gate = new object();
    study.Optimize(objective, nTrials: 200, nJobs: 8, callbacks: new Action<Study, FrozenTrial>[]
    {
        (s, t) => { lock (gate) terminatorCallback.Invoke(s, t); },
    });
    ```
- **Logging.** The level and the handler list are guarded by one lock. `Emit` copies the handler list
  and calls the handlers outside that lock, from whichever thread logs. A handler must therefore be
  thread-safe, and it must not throw: `Emit` does not catch.
  - The default `StreamLogHandler` writes to `Console.Error`, which serializes each write. A record with
    an exception is two writes, so another thread's line can land between them. The handler catches its
    own write failures, as Python's `StreamHandler` does.
  - A `StreamLogHandler` over your own `TextWriter` needs a synchronized writer (`TextWriter.Synchronized`).
- **Warnings** go through the logger, on the thread that warns.
- **The progress bar** updates under one lock per bar. Each worker's update counts once and writes its
  line whole, in order. A bar with only a timeout is not supported with `nJobs` other than 1: `Optimize`
  warns `The timeout-based progress bar is not supported with n_jobs != 1.` and runs without it, as
  Python does.

### 6.7 Heartbeat

- Only the RDB storages (SQLite and EF Core) have heartbeats, and only when a heartbeat interval is set
  (`HeartbeatInterval` in `SqliteRdbOptions` or `EFCoreRdbOptions`, Python's `heartbeat_interval`). They
  are off by default. A trial whose heartbeat is older than the grace period (twice the interval by
  default) is stale.
- Each running trial then gets a background thread (§2). It beats at once, then once per interval,
  until the objective returns. It is joined before the tell, so a finished trial is never advertised as
  running.
- A failed heartbeat write ends only that thread. It is reported in the layout of Python's
  `threading.excepthook` (`Exception in thread OptunaSharp heartbeat (trial_id=N):`, then the
  exception) on standard error, and the trial goes on without heartbeats, as in Optuna.
- With heartbeats on, every trial start first runs `FailStaleTrials`, so any worker may fail another
  worker's stale trial.
  - It writes `FAIL` straight to the storage, as Python's `fail_stale_trials` does: no tell stripe, and
    no `AfterTrial`.
  - The storage arbitrates two workers failing one trial, and the loser's `UpdateFinishedTrialError` is
    swallowed.
  - The stale-trial callback (`RetryFailedTrialCallback`, for example) runs on the worker that failed
    the trial.
- A trial whose heartbeat stopped while its objective was still running (a failed write, a long pause)
  can be failed under it. Its own tell then raises `ValueError: Cannot tell a FAIL trial.`, and
  `Optimize` raises that. Optuna raises `AssertionError('Should not reach.')` there instead (§3.3).
- `Ask` and `Tell` outside `Optimize` start no heartbeat thread. When heartbeats are on, `Ask` there warns
  `Heartbeat of storage is supposed to be used with Study.optimize.`

## 7. Storages

### 7.1 Synchronization

| Storage | One instance, many threads | Several instances or processes |
|---|---|---|
| `InMemoryStorage` | one monitor; every operation is short; records are replaced on update, never changed (§6.1) | — (one process only) |
| `JournalStorage` over `JournalFileBackend` | one monitor, held across each operation's sync and append, as in Python | the lock file around each append, on the same path and protocol as Python's, so C# and Python workers exclude each other. Reads take no lock. More in the notes below the table. |
| `JournalStorage` over a Redis backend | the same monitor | on a standalone server, Redis's atomic Lua append (`INCR` then `SET` in one script). With `useCluster: true` the two are separate commands, as in Python (see below the table). |
| SQLite `RDBStorage` (inside `CachedStorage`) | one connection under one lock | SQLite's database lock. A writer waits up to the busy timeout, 30 s by default (`SqliteRdbOptions.BusyTimeoutMs`, or the URL's `?timeout=`). |
| EF Core `RDBStorage` (PostgreSQL, MySQL) | a `DbContext` per operation and no instance lock. PostgreSQL storages built from a URL or a connection string share one `NpgsqlDataSource` (one connection pool) per connection string for the life of the process; it is never disposed. MySQL storages use MySqlConnector's pool, also one per connection string. | the database: `SELECT … FOR UPDATE` where Optuna locks rows (trial creation holds the study's row), and PostgreSQL schema creation under an advisory lock |
| `GrpcStorageProxy` | the client's trial cache under one lock, held across its remote read | the server's storage. Each call runs synchronously and holds a Kestrel pool thread until the storage returns. |
| `CachedStorage` | its own lock, then the backend's | — |

Notes:

- **The default journal lock.** `JournalFileBackend`'s default lock uses Optuna's symbolic-link
  protocol. The first time this process fails to create a symbolic link (Windows without Developer Mode,
  FAT and exFAT volumes, some network shares), it switches for good to the open-file protocol on the
  same path. The two protocols exclude each other too.
- **Waiting for the lock file.** A waiter backs off from 1 ms to 16 ms, and takes over a lock left
  unchanged for the grace period (30 s by default).
- **Redis with `useCluster: true`.** A writer that dies between the `INCR` and the `SET` leaves a gap in
  the journal. Every reader then waits on that gap forever, backing off to 10 s, as in Python.
- **The gRPC server's thread pool.** A call that waits on the storage's monitor holds its pool thread
  too. A server with many concurrent clients can therefore run out of pool threads, as §7.2 describes
  for clients (not measured).

The file backend's offset cache has its own lock, so several `JournalStorage` instances may share one
backend instance. Disposing a storage twice, by either method, does nothing. The three Redis backends
guard their close with a flag set by `Interlocked.Exchange`, so `Dispose` and `DisposeAsync` may even race
on two threads; the other storages rely on their client's idempotent close. That matters because a DI
container disposes a scoped or transient storage twice.

### 7.2 Clients that complete synchronous calls on the thread pool

- **StackExchange.Redis 3.x**, used by the `net10.0` builds of the StackExchange.Redis and NRedisStack
  backends (`net8.0` uses StackExchange.Redis 2.13.17, where this was not measured):
  - a synchronous call queues the command and parks the calling thread in `Monitor.Wait`;
  - the socket writer (`Task.Run(CopyOutAsync)`) and the reader (`Task.Run(ReadAllAsync)`) are
    thread-pool tasks.

  When parked callers fill the pool, nothing can read the reply. Above its minimum the pool adds about
  one thread a second, queued work items take the new threads in order, and the calls fail after the 5 s
  `syncTimeout`. On 32 cores, 36 parked workers passed and 40 failed. The remedies:

  - dedicated threads, as `nJobs` uses;
  - `ThreadPool.SetMinThreads`;
  - StackExchange.Redis's `ConnectionMultiplexer.SetFeatureFlag("DedicatedThreads", true)` before
    connecting. That flag is process-wide, its own documentation reserves it for support guidance, and
    it adds a reader and a writer thread per connection.
- **Grpc.Net.Client** makes its synchronous calls over asynchronous HTTP/2 loops and starves the same
  way. It has no default deadline, so a starved call waits instead of failing: 400 trials took 54–57 s at
  96 parked workers.
- **The AWS SDK**, behind the S3 artifact store and the R2 store that delegates to it, has only
  asynchronous methods, and the store waits on them synchronously. Its completions need pool threads too
  (not measured).
- **The GCS and Azure Blob SDKs**, behind those artifact stores, are called through their synchronous
  methods. Whether those need the pool was not measured.
- **Not pool-dependent:** ServiceStack.Redis (its own synchronous sockets), Microsoft.Data.Sqlite, the
  synchronous APIs of Npgsql and MySqlConnector, and the journal file backend.

The same applies without `nJobs`. A host that calls `Ask`/`Tell` from many pool threads at once starves
too: 64 concurrent `Task.Run` callers on a Redis journal finished 624 of 640 trials in 92 s, with
timeouts; with `DedicatedThreads`, 640 of 640 in 11 s.

### 7.3 Throughput

- The journal holds its monitor across every round trip of an operation, as Python's `JournalStorage`
  does. A trial with two parameters costs 19 Redis round trips (11 `GET`, 4 `SETNX`, 4 `EVALSHA`), all
  serialized. At a 1.1 ms round trip a Redis journal ran 47–49 trials/s at `nJobs` 1 and 8, and 23 at 16.
  One `JournalStorage` per worker thread (separate monitors, one journal) reached 138–153 trials/s.
- The file journal syncs every append to disk. 16 contending workers sustained about 80–150 appends per
  second on an SSD.
- Sampling is serial per sampler (§6.3).

## 8. Python bridges and the GIL

- `PythonRuntime` (in `OptunaSharp.Interop.pythonnet`) runs one embedded CPython per process. After it
  starts the engine it releases the GIL (`PythonEngine.BeginAllowThreads`), and every bridge call takes
  the GIL through `PythonRuntime.Run` (`Py.GIL()`). Every access to a `PyObject`, its `Dispose`
  included, must happen under the GIL.
- The bridges (`OptunaHubSampler`, `OptunaHubPruner`, `BoTorchSampler`, `PyCmaSampler`, `PymooSampler`)
  take the GIL inside their sampler or pruner methods, that is inside the monitor. The order is the
  monitor, then the GIL.
- `OptunaHubSampler` goes on under the GIL: it takes its private `_gate` and reads the C# study through
  the storage to rebuild its Python shadow study. Storage calls take only storage locks, which come
  after the GIL (§5.2).
- **Code that holds the GIL may call OptunaSharp** (review item P3-1, fixed 2026-11-06).
  - The problem: an objective that called suggest, report, should-prune, ask or tell inside
    `using (Py.GIL())` took the locks in the opposite order. It blocked on a monitor with the GIL held,
    while the monitor's holder, inside a bridged sampler or pruner, waited for that GIL; CPython even drops
    the GIL from a thread running Python every switch interval, so the holder could lose it mid-call. At
    `nJobs` above 1 both threads waited forever. Hangs were measured with the OptunaHub sampler (after 6
    trials) and pruner and with BoTorch; the pruner case worked before `3a75811` added the monitors.
    `Optimize(nJobs > 1)` called inside `Py.GIL()` hung the same way: its caller waited for the workers
    with the GIL held, and the workers needed it.
  - The fix is the rule CPython's C extensions follow: never block on a lock while holding the GIL. Core
    enters the four locks that come before the GIL (tell stripe, trial, sampler, pruner) through
    `Internal/MonitorGate`, and runs the `nJobs` waits for its workers through `MonitorGate.Wait`.
    `PythonRuntime.Initialize` installs `GilAwareMonitor` as the gate's hooks, also when the host started
    CPython itself. The hook tries the monitor first, so an uncontended call costs one `TryEnter`. Only
    when the monitor is contended and the thread holds the GIL (CPython's `PyGILState_Check`) does it
    release the GIL around the wait, as `Py_BEGIN_ALLOW_THREADS` does, and take it back once the monitor
    is held. The wait hook releases the GIL around the whole wait, as Python's own
    `concurrent.futures.wait` does. Without a bridge both hooks are absent and Core behaves as before.
  - Measured with a scratch probe of the shapes `GilAwareMonitorTests` pins: an objective inside
    `Py.GIL()` at `nJobs: 4` over a GIL-taking sampler, or pruner, hung after 4 of 60 trials without the
    hooks and finished all 60 in about 3 s with them; `Optimize(nJobs: 4)` called inside `Py.GIL()` hung
    after 4 of 40 trials without them and finished in 0.6 s with them. The three deterministic two-thread
    scenarios (sampler, trial and pruner monitor) deadlock by construction without the hooks; a probe of
    each hung, and finished in under 0.1 s with them.
  - Reads of the study (`Trials`, `GetTrials`, `BestTrial`) take only storage locks, which come after the
    GIL, and were always safe under it.
  - Holding the GIL across OptunaSharp calls is now safe, not useful: it still serializes the threads
    that want it, so take it only around the Python work itself.
- Python still runs one thread at a time, so bridged sampling gains nothing from `nJobs`. The OptunaHub
  bridges target `nJobs: 1`.
- **`PymooSampler`** driven from a thread other than the one that created it crashed in pymoo's
  `Algorithm.ask` with an access violation in 2 of 4 runs (review item 19b, open). Create and drive it
  on one thread. That rules out `nJobs` above 1, whose workers run on other threads, and an `Optimize`
  called from a thread other than the one that built the sampler.

## 9. Numeric kernels and random generators

- **NumSharp's kernels** are single-threaded by default (`np.multithreading` is off), which also keeps
  their summation order fixed.
  - `NUMSHARP_MULTITHREADING=1`, or `np.multithreading(true)`, lets NumSharp split large kernels over
    thread-pool threads (`Parallel.For`, up to `NUMSHARP_NUM_THREADS`, 8 by default). In NumSharp
    0.70.0 that covers only the large 1-D floating-point dot product.
  - The split changes the summation order, so the last bits of a result change and a run no longer
    matches Optuna. The split work also competes with the host's work for the pool (§7.2).
  - OptunaSharp never turns it on.
- **The linear algebra** (CMA-ES's eigendecomposition, the GP stack's Cholesky, NSGA-III's hyperplane
  solve) goes through `NumSharp.Interop.OpenBLAS`.
  - OptunaSharp sets no OpenBLAS thread count, so the library's default applies, as it does in NumPy.
  - The low bits of a result depend on that count (NumSharp measured four different answers at 1, 2, 4
    and 24 threads) and on the CPU kernel OpenBLAS dispatches.
  - To pin the count, set `OPENBLAS_NUM_THREADS` in the shell before the process starts, or call
    `OpenBlasEngine.Enable(threads: n)`. `Environment.SetEnvironmentVariable` inside the process does not
    reach the native library.
  - A byte-parity run against Python needs the same count there, set before `import numpy`.
- **Concurrency.** The sampler monitor admits one call per sampler at a time. Samplers of different
  studies can run linear algebra at the same time; OpenBLAS's own worker threads are one pool for the
  process.
- **Random generators.** NumSharp's MT19937 takes no lock, where numpy's does, and concurrent draws
  corrupt it (§6.3).
  - Each of OptunaSharp's generators belongs to one object. A sampler draws its generators under its
    monitor. The others live for one computation (a forest fit, a cross-validation split), except the
    terminator evaluators', which outlive their calls (§6.6).
  - NumSharp's `np.random` is one unguarded generator for the whole process. An objective ported from
    Python that draws from `np.random` races under `nJobs`, where Python's global `np.random` is locked.

## 10. Writing thread-safe code against OptunaSharp

- **Objectives.** Under `nJobs` the objective runs on several threads at once, so the state it shares
  must be thread-safe. Each trial gets its own `Trial` object. Draw random numbers from a generator per
  thread or per trial (`Random.Shared`, or a generator seeded from `trial.Number`), never from one shared
  `System.Random` or NumSharp's `np.random` (§9).
- **Callbacks** run on the worker thread, concurrently with other trials' callbacks. They may call
  `Stop()`. Serialize `TerminatorCallback` yourself (§6.6).
- **Custom samplers and pruners.**
  - The study already serializes its calls into one instance under the instance's monitor; do not add
    a lock for them. Calls made outside a study (a test, a tool) are not serialized.
  - Avoid static mutable state, and keep per-study state keyed by the storage object and the study id:
    study ids repeat across in-memory storages.
  - Do not tell trials of the served study from a hook (§5.3).
  - A Python-backed implementation takes the GIL inside each method (`PythonRuntime.Run`). Calls into
    OptunaSharp under the GIL are safe (§8). What stays unsafe is a lock of its own that it holds while
    it takes the GIL, if code holding the GIL can wait for that lock.
- **Custom storages.** Every operation must be safe for concurrent calls on one instance: the workers,
  the heartbeat thread and readers all call it. `TellLockFor` comes from `BaseStorage`. Make `Dispose`
  idempotent.
- **Log handlers** must be thread-safe, must not throw, and must not call OptunaSharp or take the GIL
  (§5.3).
- **Trial records.** `GetTrials(deepcopy: false)` returns the storage's own records, shared by every
  thread. Read them, never mutate them, as Optuna documents for `deepcopy=False`. `Trials` and
  `GetTrials()` return copies.
- **Many workers creating one study.** `CreateStudy(loadIfExists: true)` from many workers at once is
  safe. The storage lets one worker create the study, and the others get `DuplicatedStudyError` and load
  it, as in Python.
- **`OptunaSearchCV(nJobs: n)`** passes `n` to `Optimize`, so `n` cross-validations run at once.
  - The native sklearn port clones its estimator for each, which is safe.
  - `MLNetEstimator` shares one caller-supplied `MLContext` and its folds across every call. Concurrent
    training through one `MLContext` is not tested here, so keep `nJobs: 1` with it unless you have
    verified your pipeline.
- **Artifacts.** Uploads from concurrent trials do not interfere. Each upload gets a new artifact id (a
  GUID) and its own file or object, and one attribute write records it on the trial or study.
- **Host applications.** Do not run many studies or Ask/Tell calls on pool threads at once when the
  storage client needs the pool (§7.2). Use `nJobs` or threads of your own, raise
  `ThreadPool.SetMinThreads`, or turn on `DedicatedThreads` for Redis.
- **Dependency injection.** Samplers and pruners are registered transient by default, so every study
  gets its own instance. Storages are singletons by default: one instance is thread-safe and shares its
  connection.

## 11. Tests that pin the design

| Test class | Pins |
|---|---|
| `OptimizeConcurrencyTests` | `nJobs` draws never repeat (Random, TPE); samplers with their own state complete every trial; evolution strategies never repeat a candidate; Hyperband builds its brackets once; `Optimize` on pool threads after an `nJobs` run is not a nested call; the error rules of §3.3; workers on dedicated threads; a trial's writes leave older snapshots alone; concurrent user attributes; the `nJobs` refusal |
| `OptimizeEdgeCaseTests` | the stop, timeout and error rules of §3.3–3.4, each first observed on Optuna 5.0: timeouts; `Stop()` from an objective, a callback or another thread; `MaxTrialsCallback`; `catch`; failing callbacks and sampler hooks; unacceptable results; a nested `Optimize` in a worker; a failing final write (`[Misaligned]`); the timeout-only progress bar |
| `StudyConcurrencyEdgeTests` | two threads optimizing one study; many threads asking and telling; a consistent reader during a run; enqueued trials run once and first, also when callbacks enqueue them; two tells of one trial; one sampler shared by two studies on two threads |
| `JournalConcurrencyTests` | both lock protocols under contention; stale-lock takeover after the grace period; mixed protocols on one journal; separate storages on one study; a shared backend's offset cache; many studies on one journal; enqueued trials popped once; one trial's concurrent reports, suggestions and constraints; the create race |
| `HeartbeatTests` | heartbeat threads beat and stop; a failed write ends only its thread, reported like `threading.excepthook`; trials complete when every heartbeat write fails; `FailStaleTrials` and the retry callback; `Ask`'s warning outside `Optimize` |
| `SamplerNJobsSweepTests`, `PrunerNJobsSweepTests` | every Core sampler and pruner under `nJobs` on a journal: contiguous numbers, every trial ends as its rule dictates |
| `LazyRandomStateSeedingTests` | unseeded samplers created together never share a stream; seeded ones stay reproducible |
| `ProgressBarTests` | the bar counts every `nJobs` trial once, in order |
| `ConcurrentOrderedDictTests` | the contract of the internal concurrent collection (§5.1) |
| `StorageConcurrencyTestsBase` and its SQLite, PostgreSQL, MySQL and gRPC subclasses (Storage suite) | one storage under `nJobs: 8`, eight separate storages, enqueued pops, sixteen storages creating one study name |
| `RedisJournalConcurrencyTests` (Storage suite) | separate storages on one study; `nJobs` at four times the pool's minimum does not starve the client; dispose races |
| `StorageDisposalEdgeTests` (Storage suite) | a scoped SQLite storage, gRPC proxy or EF Core storage, disposed through both of its DI registrations, synchronously or asynchronously, closes its connection, channel or owned data source |
| `JournalMixedFleetInteropTests` (Interop suite) | C# and Python workers on one journal file (both lock protocols), on Redis, and with CMA-ES, compared with a Python-only fleet |
| `CmaEsMixedFleetInteropTests` (Interop suite) | C# and Python workers taking turns on one CMA-ES study, with and without margin, run the study Python runs alone |
| `GilAwareMonitorTests` (Interop suite) | P3-1: `PythonRuntime.Initialize` installs the gate's hooks; three deterministic two-thread deadlocks (a GIL holder against a sampler, trial or pruner monitor held by a thread that needs the GIL), recovered by `Thread.Interrupt` if they regress; an objective inside `Py.GIL()` at `nJobs: 4` over a GIL-taking sampler and pruner; `Optimize(nJobs: 4)` called inside `Py.GIL()`; four ask-and-tell threads holding the GIL |
| `OptunaHubBridgeParityTests` (Interop suite) | the real hub sampler at `nJobs: 8`; the real hub sampler and median pruner with an objective inside `Py.GIL()` at `nJobs: 4` |

No test pins these yet: the reseed's entropy, `TerminatorCallback` under `nJobs`, and the error a trial
failed under its objective raises (§3.3).

## 12. Known limits and open items

- **19b:** `PymooSampler` is bound to the thread that created it, which rules out `nJobs` above 1 (§8).
- **The 32-bit seeds** (§6.3): long `nJobs` runs and large fleets can repeat a random stream. The fix is
  to seed from at least 128 bits through `RandomState.seed(uint[])`.
- **`TerminatorCallback` under `nJobs`** (§6.6): concurrent callbacks share the evaluators' generators.
  The fix is a lock in `TerminatorCallback.Invoke`; until then, wrap it in a lock of your own.
- **`MLNetEstimator` under `OptunaSearchCV(nJobs > 1)`** is untested (§10).
- **A trial failed under its objective** (§3.3, §6.7): its tell's `ValueError` reaches the caller, where
  Optuna raises `AssertionError('Should not reach.')`. This is the divergence `[Misaligned]` already
  documents for a failing final write, but no test pins this case.
- **Journal throughput:** one monitor across all of an operation's round trips, as in Python (§7.3).
- **The Redis cluster append** (§7.1): a writer that dies between `INCR` and `SET` stalls every reader,
  as in Python.
- **No cancellation token and no Ctrl+C handling:** stop through a callback. Ctrl+C ends the process
  and leaves the trials in flight `RUNNING` (§3.4).
- **No journal heartbeat:** a trial whose process dies stays `RUNNING`, as in Python.
- **Repeated candidates** by design in some samplers (§6.3).

## 13. Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|---|---|
| The GIL serializes Python code | Monitors: sampler, pruner, trial, tell stripes, storages |
| A thread blocked on a lock or in `wait()` has released the GIL | `MonitorGate`: with a Python bridge loaded, a thread holding the GIL releases it while it waits for a tell stripe, trial, sampler or pruner monitor, or for its `nJobs` workers (P3-1, fixed 2026-11-06) |
| numpy's `RandomState` locks each draw | The sampler monitor; MT19937 itself takes no lock |
| `ThreadPoolExecutor(max_workers=n_jobs)`, threads reused within the call | OptunaSharp's own dedicated threads, reused across trials and calls (P3-4, fixed 2026-11-06) |
| `wait(FIRST_COMPLETED)`, then `f.result()` | `Task.WhenAny`, then the first failure's original exception |
| The `with ThreadPoolExecutor(...)` exit | `WaitForAllWithoutRaising` |
| Executor threads start without the caller's `contextvars`; the locale is process-wide | Workers inherit the caller's `ExecutionContext`: its culture and `AsyncLocal` values |
| The interpreter waits for executor threads before it exits | Workers are background threads: `Optimize` waits for them, a process that ends does not |
| `KeyboardInterrupt` in the objective fails the running trial (`n_jobs=1`), or the running trials finish (more workers); then it propagates | Ctrl+C ends the process; the trials in flight stay `RUNNING` |
| `study._thread_local` | `Study.ThreadLocal` (`ThreadLocal<ThreadLocalStudyAttribute>`) |
| `worker_id = prefix + threading.get_ident()` | `prefix + Environment.CurrentManagedThreadId` |
| Heartbeat on a `threading.Thread`; `threading.excepthook` reports its failure | A background `Thread`; the same report, written to standard error |
| `RandomState()` and `RandomState.seed()` take 128 bits of OS entropy for the whole key | `LazyRandomState` and `ReseedRng` take 32 bits from `RandomNumberGenerator` (open, §6.3) |
| Logging handlers take their own lock per record | Handlers are called concurrently and must be thread-safe |
| Two tells of one trial: the GIL makes one practically always win | The tell stripe guarantees it |
