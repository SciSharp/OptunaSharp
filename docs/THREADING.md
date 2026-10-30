# OptunaSharp — Threading Design

**Status:** describes `master` at `f39763d` (2026-10-30). Two items are open and marked where they
apply: the cost of the `nJobs` executor for cheap trials (review item P3-4, §3.6) and a deadlock between
the Python GIL and the sampler monitor (review item P3-1, §8).

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
5. **One sampler and one pruner per study.** Their monitors make one instance safe across the threads of
   a study, not across studies: a relative sampler given a second study raises Optuna's
   ``ValueError: `IntersectionSearchSpace` cannot handle multiple studies.``

## 2. The threads

| Thread | Started by | Lives for | Runs |
|---|---|---|---|
| The caller's | your code | — | `Optimize`. With `nJobs: 1` it runs every trial itself; otherwise it runs the submit loop and waits. `Ask` and `Tell` run on whichever thread calls them. |
| `nJobs` worker | `OptimizeLoop.StartWorker`: a `LongRunning` task, which is a new background thread | one trial | the reseed, `Ask`, the objective, `Tell`, the callbacks, the progress bar |
| Heartbeat | `HeartbeatThread.Start`, when the storage has a heartbeat interval | one trial | the trial's heartbeat, once per interval until its objective returns (§6.7) |
| Storage client I/O | StackExchange.Redis, NRedisStack, Grpc.Net.Client, the AWS SDK | the connection | socket reads and writes, on the .NET thread pool (§7.2) |
| gRPC proxy server | Kestrel | one request | the wrapped storage's operation, synchronously, on a pool thread |
| OpenBLAS | the native library | the process | LAPACK and BLAS internals (§9) |
| Python | embedded CPython | the process | every bridge call, under the GIL (§8) |

OptunaSharp starts no other threads: no timers, no `Task.Run`, no thread-pool work items. Logging and the
progress bar write on the thread that logs.

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

At most `nJobs` trials are in flight at once.

### 3.2 What a worker does

A worker is Python's `_optimize_sequential(n_trials=1, reseed_sampler_rng=True)`:

1. It sets its thread's `InOptimizeLoop` flag (§4) and puts the previous value back when it ends.
2. It reseeds the study's one sampler from OS entropy, under the sampler's monitor, so concurrent workers
   never draw the same stream. This is also why a parallel run cannot be reproduced, as in Optuna.
3. It runs one trial (`RunTrial`):
   - it fails stale trials first, when the storage has heartbeats;
   - it asks for a trial, starts the heartbeat thread and runs the objective;
   - it joins the heartbeat thread and tells the result, under the trial's tell monitor (§6.5).
4. It runs the callbacks on its own thread with a deep copy of the finished trial, calls `GC.Collect()`
   when `gcAfterTrial` is set, updates the progress bar and calls the storage's `RemoveSession`.

### 3.3 Errors

- An objective exception outside `catch` fails its trial (stored as `FAIL` and logged) and is rethrown by
  the worker. The submit loop raises it the next time it waits, after the trials still running have
  finished. It raises the original exception, not an `AggregateException`.
- When several trials failed by the time of one wait, the first is raised. The others are observed and
  dropped, so none resurfaces through `TaskScheduler.UnobservedTaskException`.
- Failures among the trials that finish after the last submission are not raised, because Python never
  calls `result()` on them. They are still stored as `FAIL` and logged.
- A failing sampler `BeforeTrial` is raised whatever `catch` holds and leaves its trial `RUNNING`, as in
  Python.

### 3.4 Stopping

- `Stop()`, the timeout and `nTrials` stop the *scheduling*, and the trials in flight finish.
- `nTrials` is exact, because the loop counts submissions. A `Stop()` from a callback, such as
  `MaxTrialsCallback`'s, comes only when its trial ends, so an `nJobs` run can end up to `nJobs - 1`
  trials past that limit, as in Python.
- The timeout is measured from one start time that the submit loop and the workers share. A worker that
  starts after the deadline runs no trial.
- `Stop()` works only on a thread inside the optimize loop, that is from an objective or a callback.
  Called from any other thread it raises Python's `RuntimeError`.
- The stop flag is a `volatile` field of the `Study` object. It stops every `Optimize` call running on
  that object, and each new `Optimize` call clears it, as Python resets `_stop_flag`.
- There is no `CancellationToken`. To cancel from outside, set a flag that a callback checks, and let the
  callback call `Stop()`.

### 3.5 Why each trial gets a dedicated thread

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

`StartWorker` therefore passes `TaskCreationOptions.LongRunning`, which makes the default scheduler
create a new thread for each task. `NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool` pins this.

### 3.6 Open: the executor's throughput (P3-4)

Python's `ThreadPoolExecutor` creates up to `n_jobs` threads and reuses them for the whole call. That
reuse was not ported. Creating a thread costs about 100 µs, serialized on the submit loop, which caps
`nJobs` near 10,000 trials/s whatever the trials do. Cheap trials at `nJobs: 8` are slower than
sequential ones (Random sampler, in memory: 4.4k–6.2k trials/s against 22k–26k at `nJobs: 1`). Trials
that take a millisecond or more do not notice it.

A replacement executor must keep five properties. Each one is a measured failure of an alternative:

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

| | Thread per trial (HEAD) | Thread pool | Pool + StackExchange.Redis `DedicatedThreads` | Pool, minimum raised by `nJobs` | OptunaSharp threads, reused across calls |
|---|---|---|---|---|---|
| Redis journal, 400 trials, `nJobs` 64 / 96 | all complete | **58/400 and 85/400, `RedisTimeoutException`** | all complete | all complete | all complete |
| Trials in flight at `nJobs: 96` (960 × 1 s, in memory; ideal 10 s) | — | **at most 34; 29.4 s** | the same as the pool | 96; 10.4 s | 96; 10.3 s |
| gRPC proxy, 400 trials, `nJobs: 96` | — | **54.5 s** | the same as the pool (the flag is Redis-only) | 0.6 s | 0.6 s |
| Delay before the host's other pool work starts, `nJobs` 64–96 | — | up to 58 s | **0.5–2.9 s** | at most 85 ms | at most 16 ms |
| Process-wide setting required | none | none | a StackExchange.Redis feature flag | `ThreadPool.SetMinThreads` | none |

A dash means the configuration was not part of that run. HEAD's thread per trial runs all `nJobs` trials
at once (128 trials of 1.5 s at `nJobs: 64` took 3.2 s, ideal 3.0 s) and kept the host's work waiting at
most 3 ms.

Throughput for cheap trials (in memory, `nJobs: 8`): the pool and reused threads tie at 7.8k–8.7k and
7.9k–8.6k trials/s, against 1.9k–6.0k for a thread per trial. For 200 calls of
`Optimize(nTrials: 8, nJobs: 8)`: threads reused across calls 813 calls/s, the pool 789, a thread per
trial 443, threads created per call 81. The choice is pending.

## 4. Per-thread state

**`Study.ThreadLocal`** is a `ThreadLocal<ThreadLocalStudyAttribute>`, one per `Study` object, the port
of Python's `study._thread_local`. It holds two fields:

- **`InOptimizeLoop`** is true while this thread runs an objective or a callback of this study. It
  refuses a nested `Optimize` on the same thread (``Nested invocation of `Study.optimize` method isn't
  allowed.``), it allows `Stop()`, and it silences `Ask`'s heartbeat warning. Because it is per thread,
  two threads may run `Optimize` on one study at the same time.
- **`CachedAllTrials`** is one coherent snapshot of the study's trials. A sampler may read it many times
  while it suggests one trial. `Ask` clears it, so every trial starts from fresh history.

**The journal's worker id.** Every journal record carries `worker_id`, which is
`<storage guid>-<managed thread id>` (Python: `threading.get_ident()`). The replay credits a record to
"this worker" only when both parts match. The threads of one process therefore compete for an enqueued
trial exactly as separate processes do: the `RUNNING` record that lands first in the journal owns the
trial, and the others skip it.

## 5. Locks

### 5.1 Inventory

| Lock | Where | Protects | Held during I/O |
|---|---|---|---|
| Tell stripe | `BaseStorage.TellLockFor(trialId)`: 64 monitors per storage instance | a tell's "still running" check, the sampler's `AfterTrial` and the final state write | yes, the storage writes |
| Trial monitor | `Trial._syncRoot`, one per `Trial` | the trial's caches; every check-then-write (report, suggest, user attribute, constraint) | yes, storage calls inside |
| Sampler monitor | `BaseSampler.SyncRoot`, one per sampler instance | every call the study makes into the sampler, and its random generator | yes, samplers read the study |
| Pruner monitor | `BasePruner.SyncRoot`, one per pruner instance | `Prune` and `FilterStudy` | yes |
| Sampler-private locks | `BaseGASampler`, `JanusSampler`, `JanusDeSampler`, `MoeaDSampler`, `SmsEmoaSampler`, `RandomEmbeddingSampler`, `OptunaHubSampler` | per-study caches | taken inside the sampler monitor |
| The GIL | CPython | Python objects | taken by the bridges inside the sampler or pruner monitor |
| Storage monitors | `InMemoryStorage`, `JournalStorage`, `CachedStorage`, the SQLite `RDBStorage`, the gRPC client's trial cache | the storage's in-memory state | the journal's and SQLite's span their round trips |
| Journal file locks | `JournalFileBackend`'s offset-cache lock (short), and the lock file (`JournalFileSymlinkLock` or `JournalFileOpenLock`) around each append | the offset cache; one append at a time across processes | the lock file spans the write and its `fsync` |
| Database locks | EF Core: `SELECT … FOR UPDATE` row locks and a PostgreSQL advisory lock for schema creation; SQLite's database lock | arbitration between storage instances and processes | yes |
| Progress bar | `ProgressBar._updateLock` | the count and the line | reads the best trial |
| Logging configuration | `OptunaLogging._lock` | the level and the handler list | no: handlers run outside it |
| Python start-up | `PythonRuntime._gate` | starting the engine | once |

Every monitor is a C# `lock`, so a thread can re-enter a monitor it already holds.

### 5.2 Lock order

Outermost first. A thread that holds a lock takes only locks further down the list:

1. the tell stripe;
2. the trial monitor;
3. the sampler monitor or the pruner monitor, never one inside the other;
4. the sampler-private locks, then the GIL (bridged samplers and pruners);
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
- **Never call OptunaSharp while holding the GIL** (§8).
- **A sampler that wraps another calls it directly.** The outer monitor already covers the inner
  sampler.
- **Log handlers must not call OptunaSharp.** They can run under a storage monitor, and warnings reach
  them the same way.

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
- Why: NumSharp's MT19937 takes no lock, where numpy's does. Two threads drawing at once produced
  identical parameters (1–7 duplicates per 400 trials at `nJobs: 16`), and the samplers' caches raced.
- Consequence: the sampling of concurrent trials is serial, while their objectives run in parallel. At
  a high `nJobs`, a slow model-based sampler (GP, SAASBO, TPE over a long history) becomes the bottleneck.
- Per-study state lives in `PerStudyState<T>`, keyed by the storage object (held weakly) and the study
  id, because study ids repeat across storages. `TrialAnchor` detects a deleted study whose id was
  reused. The class is not synchronized itself: the sampler monitor covers it.
- Random numbers come from `LazyRandomState`, MT19937, never `System.Random`. An unseeded sampler seeds
  itself from OS entropy (`RandomNumberGenerator`), so samplers created at the same moment never share a
  stream: not in one process, and not in worker processes or parallel `optunasharp ask` calls.
- Some samplers can give one point to two concurrent trials, by design: JANUS without `nJobs` lanes,
  REMBO, the genetic samplers, Grid, BruteForce and QMC. The Storage page lists them with counts.

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

- **Callbacks** run on the worker thread that ran the trial, after the trial is stored, with a deep copy
  of it. Under `nJobs` the callbacks of different trials run at the same time, so a callback must be
  thread-safe.
- **Logging.** The level and the handler list are guarded by one lock. `Emit` copies the handler list
  and calls the handlers outside that lock, from whichever thread logs. A handler must therefore be
  thread-safe.
  - The default `StreamLogHandler` writes to `Console.Error`, which serializes each write. A record with
    an exception is two writes, so another thread's line can land between them.
  - A `StreamLogHandler` over your own `TextWriter` needs a synchronized writer (`TextWriter.Synchronized`).
- **Warnings** go through the logger, on the thread that warns.
- **The progress bar** updates under one lock per bar. Each worker's update counts once and writes its
  line whole, in order.

### 6.7 Heartbeat

- A storage with a heartbeat interval (the RDB storages) gets a background thread for each running
  trial. The thread beats at once, then once per interval.
- It is joined before the tell, so a finished trial is never advertised as running.
- A failed heartbeat write ends only that thread. It is reported in the layout of Python's
  `threading.excepthook`, and the trial goes on without heartbeats, as in Optuna.
- With heartbeats on, every trial start first fails the stale trials, so any worker may fail another
  worker's stale trial.

## 7. Storages

### 7.1 Synchronization

| Storage | One instance, many threads | Several instances or processes |
|---|---|---|
| `InMemoryStorage` | one monitor; every operation is short | — (one process only) |
| `JournalStorage` over `JournalFileBackend` | one monitor, held across each operation's sync and append, as in Python | the lock file around each append, on the same path and protocol as Python's, so C# and Python workers exclude each other. A waiter backs off from 1 ms to 16 ms, and takes over a lock left unchanged for the grace period (30 s by default). Reads take no lock. |
| `JournalStorage` over a Redis backend | the same monitor | Redis's atomic Lua append (`INCR` then `SET`) |
| SQLite `RDBStorage` (inside `CachedStorage`) | one connection under one lock | SQLite's database lock, with a 30 s busy timeout |
| EF Core `RDBStorage` (PostgreSQL, MySQL) | a `DbContext` per operation and no instance lock; one `NpgsqlDataSource` per connection string for the whole process | the database: `SELECT … FOR UPDATE` where Optuna locks rows (trial creation holds the study's row), and PostgreSQL schema creation under an advisory lock |
| `GrpcStorageProxy` | the client's trial cache under one lock | the server's storage, which runs each call synchronously on a Kestrel pool thread |
| `CachedStorage` | its own lock, then the backend's | — |

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
  `syncTimeout`. On 32 cores, 36 parked workers passed and 40 failed. The remedies are dedicated threads
  (as `nJobs` uses), `ThreadPool.SetMinThreads`, or StackExchange.Redis's
  `ConnectionMultiplexer.SetFeatureFlag("DedicatedThreads", true)` before connecting. That flag is
  process-wide, its own documentation reserves it for support guidance, and it adds a reader and a writer
  thread per connection.
- **Grpc.Net.Client** makes its synchronous calls over asynchronous HTTP/2 loops and starves the same
  way. It has no default deadline, so a starved call waits instead of failing: 400 trials took 54–57 s at
  96 parked workers.
- **The AWS SDK**, behind the S3 artifact store, has only asynchronous methods, and the store waits on
  them synchronously. Its completions need pool threads too (not measured).
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
- **Rule: never call a `Trial` or `Study` method while holding the GIL.** An objective that suggests,
  reports, prunes or tells inside `using (Py.GIL())` takes the two locks in the opposite order. At
  `nJobs` above 1 it deadlocks against a worker that is inside a bridged sampler. This is review item
  P3-1, open: hangs were measured with the OptunaHub sampler and pruner and with BoTorch, and the pruner
  case worked before `3a75811` added the monitors. Release the GIL, call OptunaSharp, then take the GIL
  again.
- Python still runs one thread at a time, so bridged sampling gains nothing from `nJobs`. The OptunaHub
  bridges target `nJobs: 1`.
- **`PymooSampler`** driven from a thread other than the one that created it crashed in pymoo's
  `Algorithm.ask` with an access violation in 2 of 4 runs (review item 19b, open). Create and drive it on
  one thread.

## 9. Numeric kernels

- NumSharp's kernels are single-threaded by default (`np.multithreading` is off), which also keeps their
  summation order fixed.
- The linear algebra (CMA-ES's eigendecomposition, the GP stack's Cholesky, NSGA-III's hyperplane solve)
  goes through `NumSharp.Interop.OpenBLAS`. OptunaSharp sets no OpenBLAS thread count, so the library's
  default applies, as it does in NumPy, and `OPENBLAS_NUM_THREADS` changes it. NumSharp documents that the
  result bits depend on that count, so byte-parity runs need the same setting on both sides.
- The sampler monitor admits one call per sampler at a time. Samplers of different studies can run
  linear algebra at the same time.

## 10. Writing thread-safe code against OptunaSharp

- **Objectives.** Under `nJobs` the objective runs on several threads at once, so the state it shares
  must be thread-safe. Each trial gets its own `Trial` object.
- **Callbacks** run on the worker thread, concurrently with other trials' callbacks. They may call
  `Stop()`.
- **Custom samplers and pruners.**
  - The study already serializes its calls into one instance under the instance's monitor; do not add
    a lock for them. Calls made outside a study (a test, a tool) are not serialized.
  - Avoid static mutable state, and keep per-study state keyed by the storage object and the study id:
    study ids repeat across in-memory storages.
  - Do not tell trials of the served study from a hook (§5.3).
  - A Python-backed implementation takes the GIL inside each method and never calls OptunaSharp while
    holding it.
- **Custom storages.** Every operation must be safe for concurrent calls on one instance: the workers,
  the heartbeat thread and readers all call it. `TellLockFor` comes from `BaseStorage`. Make `Dispose`
  idempotent.
- **Log handlers** must be thread-safe and must not call OptunaSharp.
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
| `StudyConcurrencyEdgeTests` | two threads optimizing one study; many threads asking and telling; a consistent reader during a run; enqueued trials run once and first, also when callbacks enqueue them; two tells of one trial; one sampler shared by two studies on two threads |
| `JournalConcurrencyTests` | both lock protocols under contention; stale-lock takeover after the grace period; mixed protocols on one journal; separate storages on one study; a shared backend's offset cache; many studies on one journal; enqueued trials popped once; one trial's concurrent reports, suggestions and constraints; the create race |
| `SamplerNJobsSweepTests`, `PrunerNJobsSweepTests` | every Core sampler and pruner under `nJobs` on a journal: contiguous numbers, every trial ends as its rule dictates |
| `LazyRandomStateSeedingTests` | unseeded samplers created together never share a stream; seeded ones stay reproducible |
| `ProgressBarTests` | the bar counts every `nJobs` trial once, in order |
| `StorageConcurrencyTests` (Storage suite) | SQLite, PostgreSQL, MySQL and gRPC: one storage under `nJobs: 8`, eight separate storages, enqueued pops, sixteen storages creating one study name |
| `RedisJournalConcurrencyTests` (Storage suite) | separate storages on one study; `nJobs` at four times the pool's minimum does not starve the client; dispose races |
| `JournalMixedFleetInteropTests` (Interop suite) | C# and Python workers on one journal file (both lock protocols), on Redis, and with CMA-ES, compared with a Python-only fleet |

## 12. Known limits and open items

- **P3-4:** a thread per `nJobs` trial; the executor decision is pending (§3.6).
- **P3-1:** the GIL is taken after the sampler monitor; never hold it while calling OptunaSharp (§8).
- **19b:** `PymooSampler` is bound to the thread that created it (§8).
- **Journal throughput:** one monitor across all of an operation's round trips, as in Python (§7.3).
- **No cancellation token and no Ctrl+C handling:** stop through a callback (§3.4).
- **No journal heartbeat:** a trial whose process dies stays `RUNNING`, as in Python.
- **Repeated candidates** by design in some samplers (§6.3).

## 13. Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|---|---|
| The GIL serializes Python code | Monitors: sampler, pruner, trial, tell stripes, storages |
| numpy's `RandomState` locks each draw | The sampler monitor; MT19937 itself takes no lock |
| `ThreadPoolExecutor(max_workers=n_jobs)`, threads reused within the call | A dedicated thread per trial (P3-4) |
| `wait(FIRST_COMPLETED)`, then `f.result()` | `Task.WhenAny`, then the first failure's original exception |
| The `with ThreadPoolExecutor(...)` exit | `WaitForAllWithoutRaising` |
| `study._thread_local` | `Study.ThreadLocal` (`ThreadLocal<ThreadLocalStudyAttribute>`) |
| `worker_id = prefix + threading.get_ident()` | `prefix + Environment.CurrentManagedThreadId` |
| Heartbeat on a `threading.Thread`; `threading.excepthook` reports its failure | A background `Thread`; the same report, written to standard error |
| `np.random.RandomState()` seeds from OS entropy | `LazyRandomState` seeds from `RandomNumberGenerator` |
| Logging handlers take their own lock per record | Handlers are called concurrently and must be thread-safe |
| Two tells of one trial: the GIL makes one practically always win | The tell stripe guarantees it |
