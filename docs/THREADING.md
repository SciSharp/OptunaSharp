# OptunaSharp — Threading Design

**Status:** describes the code at `f39763d` (2026-10-30), plus these changes since:

- 2026-11-06: the `nJobs` executor reuses its worker threads (review item P3-4, §3.5–3.6);
- 2026-11-06: the GIL-aware monitor entry lets code that holds the Python GIL call OptunaSharp (review item
  P3-1, §8);
- 2026-11-07: the executor can be chosen at four levels, OptunaSharp's dedicated threads (the default) or the
  .NET thread pool ([plan 10](plans/10-threading.md), Phase 1, §3.5);
- 2026-11-07: OS seeds take numpy's 128-bit construction (§6.3), `TerminatorCallback` serializes its calls
  (§6.6), and the stale-trial error divergence is pinned (§3.3) (plan 10, item 2.1);
- 2026-11-07: the Redis journal reads in batches, seeds its counter inside the append script, keeps
  OptunaSharp's snapshot under a key of its own at an interval chosen by read mode (§7.3), and its
  StackExchange.Redis connections get dedicated threads by default (§7.2) (plan 10, items 2.2 and 2.3);
- 2026-11-07: `Optimize` takes a `CancellationToken` (`ct`), trials see it as `Trial.CancellationToken`, and
  the opt-in `StopSignals` turns Ctrl+C and SIGTERM into it (§3.4) (plan 10, item 2.4);
- 2026-11-07: every Python bridge call runs on one Python thread that keeps its CPython thread state for the
  life of the process (§8), which closes review item 19b, and the `nJobs` threads reserve 16 MB of stack
  (§2) (plan 10, item 2.5);
- 2026-11-07: journal records parse from their UTF-8 bytes, OptunaSharp's journal snapshot is a binary format
  (v2) about 20 times faster to save and 17 times faster to restore, trials share their distribution objects,
  and the Redis snapshot interval is Optuna's 100 in both read modes (§7.3) (plan 10, item 2.7);
- 2026-11-08: the `ConcurrencyGun` test harness fires dedicated threads into the same code at the same instant,
  and seven test classes built on it pin the guarantees of §1.5 and §3–§7 (§11);
- 2026-11-08: the GIL-aware hooks read GIL ownership from the thread's own CPython thread state, not from
  `PyGILState_Check`, which a sub-interpreter turns off for the whole process (§8; catalog Open item 28);
- 2026-11-08: three options and one fix from the second threading round ([plan 11](plans/11-threading-round-2.md)):
  `TrialExecutorOptions.SamplerConcurrency` lets each `nJobs` worker sample with an instance of its own, for the
  samplers that produce no duplicate samples that way (§6.3); `TrialExecutorOptions.BlasThreads` runs OpenBLAS
  single-threaded while several workers run (§9); and every log handler serves one record at a time under a lock
  of its own, as Python's `Handler.handle` does, with one write per record (§6.6);
- 2026-11-08: no Core sampler hands one point to two trials asked at once: the samplers that derived a candidate
  from state that changes only when a trial finishes now also look at the running trials, which leaves sequential
  runs unchanged; BruteForce diverges from Optuna's strict mode to do so; the Python bridges are measured and
  pinned (§6.3; [plan 12](plans/12-parallel-read-duplicates.md)).

No threading item is open. §12 lists the known limits.

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
| `nJobs` worker, the default executor | `OptimizeLoop.StartWorker`, on `TrialWorkerScheduler` (§3.5): a background thread of OptunaSharp's own named `OptunaSharp trial worker`, with a 16 MB stack reservation (CPython's own thread stack on macOS; §8) | until it has waited `ThreadIdleTimeout` (30 s by default) for a trial; meanwhile it runs trials of any study and any later `Optimize` call | the reseed, `Ask`, the objective, `Tell`, the callbacks, the progress bar |
| `nJobs` worker on the thread pool, opt-in (§3.5) | `OptimizeLoop.StartWorker`: a task queued to the .NET thread pool's global queue | one trial; the pool keeps the thread for its other work | the same |
| Heartbeat | `HeartbeatThread.Start`, for each trial of an RDB storage whose heartbeat interval is set (§6.7) | the trial's objective | the trial's heartbeat, once at the start and then once per interval. A background thread named `OptunaSharp heartbeat (trial_id=N)`. |
| Storage client I/O | StackExchange.Redis, NRedisStack, Grpc.Net.Client, the AWS SDK | the connection | socket reads and writes, on the .NET thread pool (§7.2) |
| gRPC proxy server | Kestrel | one request | the wrapped storage's operation, synchronously, on a pool thread |
| OpenBLAS | the native library | the process | LAPACK and BLAS internals (§9) |
| The Python thread | `PythonRuntime.Initialize`, on first use of any Python bridge: a background thread named `OptunaSharp Python`, with a 16 MB stack | the process | every bridge call (`PythonRuntime.Run` body), one at a time under the GIL, on one CPython thread state that is never freed; it also starts the embedded engine (§8) |

OptunaSharp starts no other threads: no timers and no `Task.Run`. Python's own threads, if a library starts any,
belong to CPython. The only thread-pool work items it queues
are the trials of a call that chose the thread-pool executor. Logging and the progress bar write on the
thread that logs.

**What a worker inherits.** A worker starts with the caller's `ExecutionContext`, as every .NET task does, on
either executor: its `CurrentCulture` and `CurrentUICulture`, and every `AsyncLocal<T>` value (which is also
how `Activity.Current` and logging scopes travel). Measured: the workers and callbacks of a caller running
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

- First, whatever `nJobs` is, the call resolves its executor and reads its options once (§3.5). Invalid
  values from a derived options class raise `ArgumentOutOfRangeException` here, before any trial.
- `nJobs: 1`, the default, runs `OptimizeSequential` on the caller's thread, whatever the executor.
- `nJobs: -1` uses `Environment.ProcessorCount` workers (Python: `os.cpu_count()`), on either executor.
- `nJobs` of 0 or below -1 raises `ValueError: max_workers must be greater than 0`, the refusal of
  Python's `ThreadPoolExecutor`.
- Any other value runs Python's submit loop on the caller's thread:

```text
for each submission:
    stop when Stop() was called, the timeout has passed, or nTrials trials were submitted
    when nJobs trials are in flight:
        wait for the first to finish (Task.WhenAny) and raise the first failure among the finished ones
    start a worker for one trial, on the call's executor
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
   - The new key comes from 128 bits of OS entropy, through numpy's own construction (§6.3).
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
`OptimizeConcurrencyTests` pin every row.

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
| the tell, because `FailStaleTrials` failed the trial while its objective ran (§6.7) | `FAIL` | yes: `ValueError: Cannot tell a FAIL trial.` The same divergence (`[Misaligned]`, `TrialFailedAsStaleWhileItsObjectiveRuns_RaisesCannotTellAFailTrial`): Optuna 5.0 raises `AssertionError('Should not reach.')` by the same route, with the `ValueError` as its context, and the `ValueError` itself under `python -O` (measured 2026-10-30 sequentially and at `n_jobs=2`, again 2026-11-07). The owner chose to keep the clearer error (plan 10, item 2.1). |

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
- **Cancellation** (plan 10, item 2.4, the owner's choices of 2026-11-07). `Optimize(…, ct:)` takes a
  `CancellationToken`. The submit loop and every worker check it before each trial, ahead of the stop
  flag, so a run that is both stopped and cancelled reports the cancellation.
  - Once the token is cancelled no trial starts. The trials in flight finish and are stored, and
    `Optimize` throws `OperationCanceledException` carrying the token. An already cancelled token runs
    no trial.
  - Every trial gets the token as `Trial.CancellationToken`; an asked trial has
    `CancellationToken.None`.
  - An `OperationCanceledException` that escapes an objective while the token is cancelled stores that
    trial `FAIL`, whatever `catch` holds, and the callbacks skip it. This is Python's
    `KeyboardInterrupt`, which is no `Exception` subclass, so `catch` can never hold it.
  - An objective may cancel through a token linked to the trial's. `Optimize` then reports the caller's
    token and keeps the objective's exception as the cause.
  - A cancellation exception while the token is not cancelled (an HTTP timeout, say) is an ordinary
    failure, subject to `catch`.
  - Python, observed on Optuna 5.0.0: at `n_jobs` 1 the interrupt lands in the objective, whose trial is
    stored `FAIL` (trials 0–3 COMPLETE, 4 FAIL), then `KeyboardInterrupt` propagates. At `n_jobs` 3 the
    trials in flight finish COMPLETE and the interrupt is raised after them. The token is cooperative,
    so the first outcome needs an objective that observes it.
- **Ctrl+C and SIGTERM.** .NET ends the process on either signal by default, and the worker threads end
  with it. The trials in flight stay `RUNNING` in a persistent storage: an RDB storage with heartbeats
  fails them later as stale, and a journal never does.
  - `StopSignals.Listen()` (opt-in; OptunaSharp installs no handler by itself, because a handler is
    process-wide) registers SIGINT and SIGTERM through `PosixSignalRegistration`. The first signal is
    held, and the process lives on. It also cancels `StopSignals.Token` and logs a warning. A second
    signal takes its default action, which ends the process.
  - The handlers cost nothing until a signal arrives. Measured over 10 idle seconds, a process used the
    same CPU with and without them: Windows creates the handler thread only at the signal, and Linux
    adds one runtime thread asleep in a blocking read.
  - Checked with real signals on 2026-11-07. In a Linux container, `docker kill` with SIGTERM or SIGINT
    stopped a 4-worker journal study gracefully about 1 s later: every trial COMPLETE in a new storage,
    exit code 0. Two quick SIGINTs exited with 130. On Windows, a real `CTRL_C_EVENT` on the demo's own
    console stopped it 0.4 s later, and two exited with `0xC000013A`.
  - Windows passes a parent's "ignore Ctrl+C" on to its children. A process started that way never sees
    Ctrl+C, with or without the helper: the launcher in that check had to clear it first.
  - A .NET Generic Host already handles both signals; it passes `ApplicationStopping` as the token.

### 3.5 The executor: dedicated threads by default, the thread pool on request

`StartWorker` runs each worker on the executor the call resolved. There are two
([`TrialExecutorOptions.cs`](../src/OptunaSharp.Core/TrialExecutorOptions.cs),
[`Internal/TrialExecutorSnapshot.cs`](../src/OptunaSharp.Core/Internal/TrialExecutorSnapshot.cs)):

- **`TrialExecutorKind.DedicatedThreads`, the default.** The worker is queued on a `TrialWorkerScheduler`:
  OptunaSharp's own threads, never pool threads. They are created on demand and reused by later trials and
  later `Optimize` calls. Each one ends after waiting `ThreadIdleTimeout` for a trial: 30 s by default, and
  never with `Timeout.InfiniteTimeSpan`. They are background threads, so they never keep the process alive.
  - **One thread set per idle timeout.** The process keeps one scheduler for each distinct timeout. Calls
    with equal timeouts share threads, whatever study or options object they come from. The default 30 s
    entry is `TrialWorkerScheduler.Shared`.
  - **`nJobs` trials really in flight.** A queued trial gets a new thread whenever fewer workers are idle
    than trials are queued.
  - **No inlining.** A trial is never run on the thread that waits for it.
- **`TrialExecutorKind.ThreadPool`, opt-in.** Each worker is a task on the .NET thread pool, queued with
  `PreferFairness` to the pool's global queue. That has two effects:
  - An `Optimize` called from a pool thread never runs one of its own trials on that thread. The TPL runs
    a waited-on task inline only when it can pop the task from the waiting thread's own local queue, and
    a task on the global queue is not there.
  - Queued trials take turns with the host's other pool work instead of jumping ahead of it.

  `ThreadIdleTimeout` does not apply: the pool manages its own threads.

Everything else is the same on both executors:

- Python's submit and wait loop, and its error and stop rules (§3.1–3.4);
- the GIL-aware waits (§8);
- the caller's `ExecutionContext`, which every worker inherits (§2);
- the per-thread hygiene of §4: each worker restores `InOptimizeLoop` and drops its `CachedAllTrials`
  snapshot when it finishes. That matters even more on pool threads, which the host reuses.

`nJobs` keeps Python's meaning on both, `-1` included, and `nJobs: 1` runs on the caller's thread whatever
the executor.

**Choosing.** A `TrialExecutorOptions` object holds the choice in two settings, `Executor` and
`ThreadIdleTimeout`, and two more that shape the call's workers rather than their threads:
`SamplerConcurrency` (§6.3) and `BlasThreads` (§9). The class is mutable, has a public parameterless
constructor, and is not sealed. Each `Optimize` call uses the first of four levels that is set:

1. the call's `executorOptions` argument;
2. the study's `ExecutorOptions` property. It is `null` until set. It belongs to the `Study` object, not to
   the stored study, so a study opened again (`LoadStudy`, `CreateStudy(loadIfExists: true)`) starts with
   `null`;
3. the DI default, `OptunaStudyOptions.DefaultExecutorOptions`. `OptunaStudyFactory` assigns it to the
   `ExecutorOptions` of every study it builds, so a value set on the study afterwards overrides it;
4. the process-wide `TrialExecutorOptions.Default`: dedicated threads with a 30 s idle timeout, unless the
   application replaces it. Assigning `null` throws `ArgumentNullException`.

Optuna has no counterpart: Python always runs `ThreadPoolExecutor(max_workers=n_jobs)`.

**The snapshot.** When the call starts, the winning object's four getters (`Executor`,
`ThreadIdleTimeout`, `SamplerConcurrency`, `BlasThreads`) are each called exactly once. A derived class's
overrides are therefore honored, and a computed value is computed once. Changing an options object, or
replacing `Default`, affects later calls and never a running one.

The values are validated at that point, as the setters validate theirs. An undefined `Executor` or
`SamplerConcurrency` raises `ArgumentOutOfRangeException` before any trial, and so does a `BlasThreads` of
zero or less (`null` is the automatic policy) and a `ThreadIdleTimeout` that is any of:

- zero;
- negative, other than `Timeout.InfiniteTimeSpan`;
- longer than `int.MaxValue` milliseconds, the longest wait `Monitor.Wait` accepts.

These are .NET exceptions, not Python ones, because the settings have no Optuna counterpart. The setters
and `Default` may be used from any thread: they use volatile reads and writes. The timeout is stored as
64-bit ticks, so a 32-bit process never reads a torn value; `SamplerConcurrency` and `BlasThreads` are stored
as 32-bit integers (0 encodes a null `BlasThreads`, which no valid count can be).

**The pool's limits are documented, not handled** (the owner's decision, 2026-11-07). A trial blocks its
thread for the whole trial. Once `nJobs` reaches the pool's minimum worker count (`ThreadPool.GetMinThreads`,
the processor count by default), the pool adds threads only about once a second. Measured in §3.6:

- clients that complete their synchronous calls on the pool stall or time out (StackExchange.Redis,
  Grpc.Net.Client, the AWS SDK; §7.2);
- fewer trials run at once than `nJobs` asks for: at most 33 at `nJobs: 96` on 32 cores, so 192 trials of
  1 s took 6.1 s instead of 2 s;
- the host's other pool work waits behind the queued trials, up to 2 s in that run.

OptunaSharp does not check for this, warn, or raise the minimum. If it happens, use `DedicatedThreads`,
which reuses OptunaSharp's own threads, or raise the pool's minimum with `ThreadPool.SetMinThreads`.

**Why dedicated threads are the default.** Until `3a75811` the workers ran on the .NET thread pool
(`Task.Run`). That caused two failures:

- **Pool starvation.** A worker is blocked for most of a trial: on the storage's lock, on file or
  network I/O, in the objective. With `nJobs` above the pool's minimum thread count every pool thread was
  blocked, and the clients that complete their synchronous calls on the pool stalled (§7.2).
  `Optimize(nJobs: 96)` on a StackExchange.Redis journal died after 88 trials with
  `RedisTimeoutException`.
- **Stale thread-local state (review item H5).** A pool thread outlives the call, so a worker's
  `InOptimizeLoop = true` stayed on the thread: 61 of 64 later `Optimize` calls on pool threads failed
  with a false "Nested invocation" error. Workers now restore the flag (`restoreLoopFlag`), whatever
  thread runs them.

`3a75811` then gave every trial a new thread (`TaskCreationOptions.LongRunning`), which made cheap trials
slow (§3.6). Since 2026-11-06 the trials reuse the dedicated threads described above. The opt-in pool
executor of 2026-11-07 keeps the H5 fix, because the fix lives in the worker, not in the executor.

**Pinned by tests:**

- `NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool`;
- `TrialWorkerSchedulerTests`: reuse, `nJobs` really in flight, no inlining, retirement;
- `TrialExecutorOptionsTests`: both executors, the four levels, the snapshot, validation, the idle-timeout
  thread sets, and Python's rules on the pool (§11).

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

| | Thread per trial (the executor then) | Thread pool | Pool + StackExchange.Redis `DedicatedThreads` | Pool, minimum raised by `nJobs` | OptunaSharp threads, reused across calls |
|---|---|---|---|---|---|
| Redis journal, 400 trials, `nJobs` 64 / 96 | all complete | **58/400 and 85/400, `RedisTimeoutException`** | all complete | all complete | all complete |
| Trials in flight at `nJobs: 96` (960 × 1 s, in memory; ideal 10 s) | — | **at most 34; 29.4 s** | the same as the pool | 96; 10.4 s | 96; 10.3 s |
| gRPC proxy, 400 trials, `nJobs: 96` | — | **54.5 s** | the same as the pool (the flag is Redis-only) | 0.6 s | 0.6 s |
| Delay before the host's other pool work starts, `nJobs` 64–96 | — | up to 58 s | **0.5–2.9 s** | at most 85 ms | at most 16 ms |
| Process-wide setting required | none | none | a StackExchange.Redis feature flag | `ThreadPool.SetMinThreads` | none |

A dash means the configuration was not part of that run. The thread per trial, the executor then, ran all
`nJobs` trials at once (128 trials of 1.5 s at `nJobs: 64` took 3.2 s, ideal 3.0 s) and kept the host's work
waiting at most 3 ms.

Throughput for cheap trials (in memory, `nJobs: 8`): the pool and reused threads tie at 7.8k–8.7k and
7.9k–8.6k trials/s, against 1.9k–6.0k for a thread per trial. For 200 calls of
`Optimize(nTrials: 8, nJobs: 8)`: threads reused across calls 813 calls/s, the pool 789, a thread per
trial 443, threads created per call 81. Threads reused across calls were chosen (2026-11-06): they are the
only option that keeps all five properties and matches the pool's throughput. The pool fails properties 4
and 5, so it can only ever be an opt-in. The owner decided on 2026-11-07 to expose it as an opt-in through
an options class at four levels (a static default, a DI default, a study setting, a per-call argument),
with the plain pool's limits documented rather than handled. It was built the same day (§3.5;
[plan 10, Phase 1](plans/10-threading.md#phase-1--the-njobs-executor)).

The built executors, re-measured on 2026-11-07 on the same 32-CPU machine (pool minimum 32). The run was
192 trials of 1 s at `nJobs: 96` (ideal 2 s), with a host thread queuing a pool work item every 50 ms:

| | `DedicatedThreads` (default) | `ThreadPool` (opt-in) |
|---|---|---|
| Trials in flight, at most | 96 | 33 |
| Wall time | 2.0–2.3 s | 6.1 s |
| The host's pool work waited, at most | 1 ms | 2.0 s |

## 4. Per-thread state

**`Study.ThreadLocal`** is a `ThreadLocal<ThreadLocalStudyAttribute>`, one per `Study` object, the port
of Python's `study._thread_local`. It holds two fields, and a third of OptunaSharp's own:

- **`InOptimizeLoop`** is true while this thread runs an objective or a callback of this study.
  - It refuses a nested `Optimize` on the same thread (``Nested invocation of `Study.optimize` method
    isn't allowed.``), it allows `Stop()`, and it silences `Ask`'s heartbeat warning.
  - Because it is per thread, two threads may run `Optimize` on one study at the same time.
  - It is also per `Study` object, as in Python: a second `Study` object loaded for the same study has
    flags of its own.
- **`CachedAllTrials`** is one coherent snapshot of the study's trials. A sampler may read it many times
  while it suggests one trial. `Ask` and every tell clear it, so every trial starts from fresh history.
- **`SamplerOverride`** is the worker's own sampler instance while the thread runs a trial of an
  `Optimize(nJobs)` call with `SamplerConcurrency.PerWorker` (§6.3), and null otherwise. A `Trial` created on
  the thread binds it, and the tell on the thread reaches it (`Study.SamplerForCurrentThread`).

**Reused threads.** A worker's thread outlives its trial on both executors (§3.5). A dedicated thread runs
later trials of any study, and a pool thread goes back to the host's other work. Each worker therefore
leaves its thread as it found it:

- it puts back the `InOptimizeLoop` value it found (`restoreLoopFlag`);
- it drops its `CachedAllTrials` snapshot;
- it puts back the `SamplerOverride` it found and returns its sampler instance to the call's pool, so a
  reused thread never samples a later trial with an instance from an earlier call
  (`PerWorker_LeavesNoWorkerInstanceOnTheReusedThreads`).

The next trial, or the host's next work item, then sees no state of this study. On the pool this is pinned
by `ThreadPool_AfterARun_PoolThreadsCarryNoOptimizeLoopState`: a later `Optimize` on any pool thread is not
taken for a nested call.

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
| Log handler locks | `OptunaLogging._handlerLocks`, one per handler instance (Python's `Handler.lock`) | one record at a time per handler | yes: the handler's write |
| Sampler worker pool | `SamplerWorkerPool._gate`, one per `Optimize` call with `SamplerConcurrency.PerWorker` | renting and returning worker instances | no |
| OpenBLAS thread count | `BlasThreadPolicy._gate` (static) | the counts held by running calls, and the native setting | no; one native call when the smallest count changes |
| Python start-up | `PythonRuntime._gate` | starting the engine | once |

Every monitor is a .NET monitor, so a thread can re-enter a monitor it already holds. The four that come
before the GIL in the lock order (§5.2) are entered through `MonitorGate` instead of a C# `lock`: the tell
stripes, the trial monitor, and the sampler and pruner monitors. Without a Python bridge that is a plain
`Monitor.Enter`; with one, a thread that holds the GIL releases it while it waits (§8). The others stay
plain `lock`s, because no holder of theirs ever waits for the GIL.

**Shared state without a lock.** A few fields are shared across threads by design, each safe on its own
terms:

- `Study.StopFlag`, a `volatile` bool (§3.4).
- The executor settings (§3.5): `TrialExecutorOptions.Default`, `Study.ExecutorOptions` and the two
  properties of a `TrialExecutorOptions` are read and written with `Volatile`. The idle timeout is kept as
  64-bit ticks, so it is never torn. A call reads each value once and keeps it for its whole run. A change
  made while a call starts can reach one of the two values and not the other, and each is valid on its own.
- Caches built on first use and published whole through one reference, so a racing reader sees the
  finished object or builds an identical one:
  - `CategoricalDistribution`'s index of its choices, and `HyperbandPruner`'s bracket plan (both
    `volatile`);
  - the static QMC tables: the Sobol' direction numbers and the Halton primes.
- `ConcurrentDictionary` caches:
  - `TrialWorkerScheduler`'s schedulers, one per idle timeout (static; a racing first use may build two,
    and the one that loses has started no thread);
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
4. inside a sampler: its private locks, and for the Python bridges the Python thread and the GIL it
   takes there: the caller waits for the Python thread, which runs the body under the GIL (§8).
   `OptunaHubSampler` goes the furthest: the monitor, then the Python thread and the GIL, then its private
   `_gate` (taken only on the Python thread), then the storage, which it reads to mirror the study into
   its Python shadow study;
5. the storage monitor: `CachedStorage`'s before its backend's, `JournalStorage`'s before its backend's
   offset-cache lock and lock file;
6. the database and cross-process locks.

The progress bar takes its lock before it reads the study. Log handlers are called outside the logging
lock, but on the thread that logs, which may hold any of the locks above: a journal storage logs a new
study from inside its monitor. Each handler runs under its own lock, the last in the order: its holder runs
only the handler, which takes no lock and not the GIL (§5.3). It is a plain monitor on purpose, not a
`MonitorGate` entry: a storage monitor's holder logs, and a GIL-releasing entry would make that holder wait to
take the GIL back, which no storage-lock holder may do.

Three locks stand outside the order because their holders take no other lock: the worker pool's `_gate`
(taken before a trial and after it, never inside one) and `BlasThreadPolicy._gate` (taken when a call starts
and ends, around one native call to OpenBLAS).

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
  surfaces in the operation that logged: a storage call, a tell, the optimize loop. Other threads that log
  wait for the handler meanwhile (its lock), so a slow handler slows every thread that logs.

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
- Why: Python runs one sampler call at a time under the GIL, and numpy's `RandomState` locks each draw.
  .NET threads run in parallel, so the monitor supplies the GIL's half: a sampler's call is many draws
  plus updates to its caches and state, and only the monitor keeps that whole call one at a time.
  - Without it, the samplers' caches raced, and two threads drawing at once produced identical
    parameters (1–7 duplicates per 400 trials at `nJobs: 16`).
  - Before NumSharp 0.80.0 the generator itself took no lock either: one generator drawn by 8 threads
    at once (200,000 draws each) repeated 30,000–40,000 values and threw 1–11
    `IndexOutOfRangeException`s per run (measured 2026-10-30). Since 0.80.0 NumSharp holds the bit
    generator's lock for every draw, as numpy does, and the same volley repeats nothing
    (`ConcurrencyGunTests.NumSharpGenerator_DrawnWithoutTheSamplerMonitor_NeverTears`).
- Consequence: the sampling of concurrent trials is serial, while their objectives run in parallel. At
  a high `nJobs`, a slow model-based sampler (GP, SAASBO, TPE over a long history) becomes the bottleneck.
- Per-study state lives in `PerStudyState<T>`, keyed by the storage object (held weakly) and the study
  id, because study ids repeat across storages. `TrialAnchor` detects a deleted study whose id was
  reused. The class is not synchronized itself: the sampler monitor covers it.
- Random numbers come from `LazyRandomState`, MT19937, never `System.Random`. An unseeded sampler seeds
  itself from OS entropy (`RandomNumberGenerator`) rather than the clock. Samplers created at the same
  moment therefore draw different streams: in one process, in worker processes, and in parallel
  `optunasharp ask` calls. On the clock, 8 parallel shells had repeated 11 and 15 of their 80 draws.
- **128 bits of OS entropy, numpy's construction** (plan 10, item 2.1, 2026-11-07). An unseeded sampler's
  generator and every `ReseedRng` take their state from `Internal/OsEntropySeed`, a port of numpy 2.4.2's
  `MT19937.__init__` and `MT19937._legacy_seeding(None)`:
  - a `SeedSequence` of 128 fresh bits is expanded by `generate_state(624)` into the whole key, and the
    first word is forced to `0x80000000`;
  - a new generator starts at position 623 (numpy keeps its key-copy loop's last index), so its first draw
    tempers the key's last word before the first twist;
  - a reseed keeps the position where it was and drops the cached Gaussian, as numpy's `seed()` does;
  - an explicit seed overwrites this state at once (`LazyRandomState(seed)` calls `seed(seed)` on the new
    generator, as Python's does), so seeded runs are unchanged.

  Until then every OS seed was one 32-bit word through `RandomState.seed(uint)`. Two of N such seeds are
  equal with probability about N²/2³³: 1% at 9,300 seeds, 50% at 77,000. A generator reseeded with a seed
  it had before replays the earlier stream, so an `nJobs` run, which reseeds once per trial, could hand a
  later trial an earlier trial's `RandomSampler` point or random-startup values.
  - Measured 2026-10-30: one million `RandomSampler.ReseedRng` calls, each followed by one draw, repeated
    116 draws, where the birthday bound predicts 116.4; one million numpy reseeds repeated none.
  - Measured 2026-11-07, 300,000 reseeds: the old path repeated 9–14 draws in five rounds (bound 10.5),
    the port none. `OsEntropySeedTests` pins both the construction, against numpy output for injected
    entropy, and the absence of repeats.
  - `BoTorchSampler.ReseedRng` draws a new 31-bit integer seed for torch, as Python's
    `RandomState().randint(int32 max)` does, so it never had a key to fill.
- **Trials asked at once get different points** ([plan 12](plans/12-parallel-read-duplicates.md), 2026-11-08).
  Eight trials that ask and suggest on the same history before any is told (a *parallel read*) receive eight
  different points from every Core sampler they share in one process. The samplers that derived a candidate from
  state that changes only when a trial finishes now also look at the other running trials (collisions are trials
  of a round that received a point another trial of the round received, 8 rounds of 8, before the change):

  | Sampler | Collisions before | What a trial asked while others run gets now |
  |---|---|---|
  | `JanusSampler`, `JanusDeSampler` | 7 (and 30, 29 of 120 under `Optimize(nJobs: 16)`) | a slot of the generation no trial received yet, running or finished; with none left, an independent sample |
  | `PsoSampler`, `DifferentialEvolutionSampler` | 1–9, up to 2 | the first particle or target no finished or running trial of the generation holds (stamped in the storage), and a point no running trial holds (redrawn up to 32 times); otherwise an independent sample, which is not warned about |
  | `LmCmaSampler` | 8–10 | a Rademacher candidate no running trial of the generation holds (redrawn up to 32 times); otherwise an independent sample |
  | `TurboSampler`, `SaasboSampler` | every startup round, 7 | the first row of the startup Sobol design no completed or running trial holds (stamped `turbo:init`, `saasbo:init`); rows past the design are the sequence's next points |
  | `RandomEmbeddingSampler` | up to 21 | an acquisition fitted with the running trials' latents as constant lies, so its optimum moves |
  | NSGA-II/III and their CPT variant, AGE-MOEA-2 and its CPT variant, SMS-EMOA, RVEA, MOEA/D, the JANUS multi-objective samplers | 1 in one run of 12–20 | a child generated again while a running trial holds it (up to 32 times); otherwise an independent sample |
  | `BruteForceSampler`, both modes | strict 3–8 per 88 trials, default 0–1 | a point no running trial holds or will take (below) |

  - **Sequential runs are unchanged.** Every rule but JANUS's keys on *other* running trials, and a run of one trial
    at a time has none; JANUS's hands each slot of a generation out once, which consecutive trials asked one at a time
    do anyway. So such a run never skips a slot, redraws or lies, and seeded results, Optuna parity and JANUS's byte
    parity with its reference are the same. (One edge: after a JANUS trial fails, its generation is never told, and
    the trials past its slots are sampled independently instead of evaluating its candidates again.)
  - **Pinned by** `SamplerParallelReadTests`, over every Core sampler, both on eight threads (`ParallelReads.Fire`)
    and in lockstep, every read of a round landing before any write (`ParallelReads.ReadInLockstep`), with the
    per-worker instances too; with populations larger than a round, so a generation's trials finish before the
    generation is complete; and in whole `Optimize(nJobs: 16)` runs, where no sampler that never re-proposes a
    finished point evaluated a point twice in 120 trials (xNES aside: its distribution can degenerate one trial at
    a time too, plan 12 §8). The rounds alone were not enough: a first JANUS fix,
    which freed a slot once its trial finished, passed every volley and still repeated 41–65 points in 120 trials
    of `Optimize(nJobs: 16)`, because rounds release a population of 7 together.
  - **History repeats are not collisions.** The genetic samplers, DE, JANUS-DE, PSO and LM-CMA re-propose finished
    points by design (a child that copies a parent, a mutant clamped onto a corner, a particle at rest, a Rademacher
    point of an earlier generation), one trial at a time as well; every other Core sampler repeats no told trial
    under parallel reads either, except that CMA-ES and VkD-CMA seed each trial with `randint(1, 65536) +
    trial.number`, as Optuna does, so two trials of a generation share a point about once in 65,535 pairs.
  - **One process.** PSO's particles, DE's targets and TuRBO's and SAASBO's design rows are stamped in the storage,
    so they steer apart asks from different processes too. JANUS's slots, the redraws and BruteForce's record of
    handed-out values live in the sampler instance, guarded by its monitor: two processes on one storage can still
    receive one point.
  - **BruteForce diverges from Optuna** (`[Misaligned]`, the owner's decision). It samples one parameter per call, so
    it records the values handed to each running trial, counts a trial still choosing as taking one point below its
    path, and enters a branch whose points are all held only when every branch is. Optuna's strict mode
    (`avoid_premature_stop=True`) hands running points out again on purpose, so a dead worker's point is still
    evaluated; here it is not, until the dead trial is failed. When running trials hold nearly every unexplored
    point, a trial re-evaluates a finished point instead.
  - **Grid is unchanged:** it hands out the ids no finished or running trial holds, as Optuna does, and once every id
    is taken, any id, a running one included, until the study stops.
  - **The Python bridges are unchanged:** each hands out what the Python code it drives hands out.
    `BridgeSamplerParallelReadTests` pins them (8 rounds of 8; collisions in lockstep, then on threads): PyCma 49
    (exactly `optuna_integration.PyCmaSampler`'s points, compared bit for bit), 0–1; pymoo NSGA-II and AGE-MOEA-2
    42, 0–2, and on threads it re-hands 7–8 offspring that told trials evaluated; OptunaHub `random_search` and
    `pso` 0, 0; its `differential_evolution` 49, 49; BoTorch 0, 0.

**Per-worker sampler instances** (opt-in, [plan 11](plans/11-threading-round-2.md), 2026-11-08).
`TrialExecutorOptions.SamplerConcurrency` chooses how the workers of one `Optimize(nJobs)` call reach the
sampler:

- **`Shared`, the default, is Optuna's model.** Python's `n_jobs` threads share the study's one sampler object,
  and the sampler monitor above makes its calls one at a time, so the sampling is serial.
- **`PerWorker` gives each worker an instance of its own** for the samplers that take part. The call keeps a
  `SamplerWorkerPool`. Before its trial a worker rents an instance made from the study's sampler (read once,
  so a sampler swapped onto the study mid-run gets instances of its own), reseeds it from fresh OS entropy as
  Optuna reseeds a worker, and binds it to the thread (`SamplerOverride`, §4). The `Trial` created on that
  thread keeps the instance for all its calls, so `BeforeTrial`, the sampling and the tell's `AfterTrial` all
  reach the instance whose state sampled the trial. After the trial the worker returns the instance for a later
  trial of the call; a call holds at most `nJobs` instances. Each instance has its own monitor, so the workers
  sample in parallel, exactly as separate processes on one storage do. `nJobs: 1` ignores the setting.
- **Which samplers take part.** A sampler takes part by returning an instance from the internal
  `BaseSampler.CreateWorkerInstance`; the others return null and keep the shared instance, one call at a time,
  and the pool remembers that after the first ask. A sampler written outside OptunaSharp cannot override the
  member, so it always stays shared; so does a subclass of a taking-part sampler, whose copy would drop what the
  subclass adds. The rule: a sampler takes part only when eight per-worker instances on one study never
  proposed the same point twice in any round. Measured 2026-11-08, three rounds of 64 trials (24 for GP, TuRBO
  and REMBO, 16 for SAASBO, 90 for Grid and BruteForce), repeated points per round:

  | Sampler | Shared (one instance) | Per worker | Takes part |
  |---|---|---|---|
  | `RandomSampler`, `TPESampler` (both modes), `CmaEsSampler`, `VkdCmaSampler`, `GridSampler`, `PartialFixedSampler` over TPE | 0, 0, 0 | 0, 0, 0 | yes |
  | `QMCSampler` (Sobol', scrambled or not) | 0, 0, 0 | 4–20 | no: every instance walks the same sequence |
  | `OpenAiEsSampler`, `SnesSampler`, `XnesSampler` | 0, 0, 0 | 21–29 | no: every instance starts the same generation |
  | `JanusSampler`, `JanusDeSampler` | 6–18 | 49–54 | no |
  | `PsoSampler`, `TurboSampler`, `RandomEmbeddingSampler`, `SaasboSampler`, `BruteForceSampler` | 0–7 | 2–32 | no |
  | `GPSampler` | 0, 0, 0 | 0, 1, 0, then 0 in eight more rounds | no: instances that see the same history optimize the same acquisition function |
  | `LmCmaSampler`, the genetic and many-objective samplers (NSGA-II/III, SMS-EMOA, AGE-MOEA-2, RVEA, MOEA/D, DE, the JANUS hosts) | 0–11 (by design, above) | 0–15 | no: they repeat points in both modes, so per-worker gains nothing safe |

  The "Shared" column was measured before plan 12 gave every Core sampler different points for trials asked at
  once (above); the decision it informed stands, because a per-worker instance sees neither the shared instance's
  record of handed-out points nor, for the samplers that redraw, the other instances' draws.
  `OnlyTheListedSamplers_MakeWorkerInstances` pins the list, and
  `TakingPartSamplers_UnderPerWorker_ProposeNoPointTwice` pins the "no repeat" property for each of them at
  `nJobs: 8` (CMA-ES and VkD-CMA tolerate one repeat: each trial seeds the optimizer with a draw from 1 to
  65535 plus the trial number, so two trials of one generation repeat a point when those collide, about once in
  65,535 pairs, in both modes, as in Optuna).
- **What each instance shares.** Only immutable configuration (parameters, the grid, `x0`, warm-start
  trials). Generators, search-space caches and model state are new per instance, and the CMA family's
  independent sampler is copied too (a CMA-ES whose independent sampler cannot be copied stays shared).
- **Footguns.** Functions handed to a taking-part sampler (TPE's `gamma`, `weights`, `constraintsFunc`) run on
  several workers at once and must be thread-safe. The study's own sampler instance is not used by such a call's
  workers, so caches it keeps in memory are not updated by them. A trial asked inside an objective and told on
  another thread is told to the study's sampler, not to the worker instance that sampled it.
- **What it buys.** The shared monitor caps throughput at one proposal per sampling time, so `PerWorker`
  pays only when sampling, not the objective, limits the run (`PerWorker_SamplesOnEveryWorkerAtOnce_EachWithItsOwnInstance`
  pins that the workers do sample at once). Measured 2026-11-08 at `nJobs: 8` on 32 cores, best of three:

  | Run | `Shared` | `PerWorker` |
  |---|---|---|
  | multivariate TPE, 10 parameters, 600 trials, 5 ms objective | 1.95 s | 1.20 s |
  | CMA-ES, 40 dimensions, 2,000 trials, no sleep | 3.68 s | 1.46 s |
  | univariate TPE, 3 parameters, 400 trials, 20 ms objective | 1.56 s | 1.54 s |

  The slowest samplers to propose do not take part: `GPSampler` ran 120 trials of a 20 ms objective in 18.7 s
  at `nJobs: 8` against 12.1 s at `nJobs: 1`, and stays so.

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
- **`TerminatorCallback` serializes its calls** (plan 10, item 2.1, 2026-11-07). Its terminator's improvement
  evaluators (`RegretBoundEvaluator`, `EMMREvaluator`) each keep one `LazyRandomState` across calls, so at
  `nJobs` above 1 concurrent callbacks used to draw from one unguarded generator (§6.3). In Python,
  numpy's lock makes the same draws safe.
  - `Invoke` now runs `ShouldTerminate` under one lock per callback, entered through `MonitorGate`. It
    comes before the GIL in the lock order, so a caller that holds the GIL releases it while it waits
    (§8), and a terminator that takes the GIL cannot deadlock. `Study.Stop` runs outside the lock.
  - Measured before the fix at `nJobs: 8`: eight callbacks ran at once, and 8 rounds of real terminators
    showed no failure, likely because the draws are a small part of each GP fit. The test
    `TerminatorCallback_UnderNJobs_RunsItsTerminatorOneCallAtATime` counts overlapping calls: 1 with the
    lock, 8 without it.
  - Wrapping the callback in a lock of your own is no longer needed.
- **Logging.** The level and the handler list are guarded by one lock. `Emit` copies the handler list
  and calls the handlers outside that lock, from whichever thread logs, each under **a lock of its own**
  (2026-11-08): Python's `Handler.handle` holds the handler's `lock` around `emit`, so one handler serves one
  record at a time while different handlers serve different threads at once. A handler needs no lock for what
  only its `Emit` touches, and it must not throw: `Emit` does not catch.
  - Why: the handlers were not safe without it. Measured with 8 threads logging 5,000 failed-trial warnings
    each (40,000 records, each with an exception): over `Console.Error` only 3,698 records arrived whole (the
    line followed by its own exception); over a plain `StreamWriter`, as the CLI's `--log-file` builds it, 412
    did and 48,247 lines were mangled. With the lock and the single write below, 40,000 of 40,000 in both.
  - The lock is a plain monitor, last in the lock order (§5.2), and its key is the handler instance
    (`ConditionalWeakTable`), so a handler removed and added again while another thread is still inside it
    keeps the same lock.
  - The default `StreamLogHandler` writes each record in one write, line and exception together, then
    flushes, as Python's `StreamHandler.emit` does since issue 35046. It catches its own write failures, as
    Python's `StreamHandler` does.
  - A `StreamLogHandler` over your own `TextWriter` needs no synchronized writer any more, unless other code
    writes to the same writer (then `TextWriter.Synchronized`).
  - Adding a handler that is already registered does nothing, as Python's `Logger.addHandler` ensures; before,
    it was registered twice and received every record twice.
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
  `Optimize` raises that. Optuna raises `AssertionError('Should not reach.')` there instead, a pinned
  divergence (§3.3).
- `Ask` and `Tell` outside `Optimize` start no heartbeat thread. When heartbeats are on, `Ask` there warns
  `Heartbeat of storage is supposed to be used with Study.optimize.`

## 7. Storages

### 7.1 Synchronization

| Storage | One instance, many threads | Several instances or processes |
|---|---|---|
| `InMemoryStorage` | one monitor; every operation is short; records are replaced on update, never changed (§6.1) | — (one process only) |
| `JournalStorage` over `JournalFileBackend` | one monitor, held across each operation's sync and append, as in Python | the lock file around each append, on the same path and protocol as Python's, so C# and Python workers exclude each other. Reads take no lock. More in the notes below the table. |
| `JournalStorage` over a Redis backend | the same monitor | on a standalone server, Redis's atomic Lua append (the counter's `SETNX`, then `INCR` and `SET`, in one script). With `useCluster: true` the commands are separate, as in Python (see below the table). |
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
  backends (`net8.0` uses StackExchange.Redis 2.x; see below):
  - a synchronous call queues the command and parks the calling thread in `Monitor.Wait`;
  - the socket writer (`Task.Run(CopyOutAsync)`) and the reader (`Task.Run(ReadAllAsync)`) are
    thread-pool tasks.

  When parked callers fill the pool, nothing can read the reply. Above its minimum the pool adds about
  one thread a second, queued work items take the new threads in order, and the calls fail after the 5 s
  `syncTimeout`. On 32 cores, 36 parked workers passed and 40 failed. The remedies:

  - dedicated threads, as `nJobs` uses by default (`TrialExecutorKind.DedicatedThreads`, §3.5);
  - `ThreadPool.SetMinThreads`;
  - StackExchange.Redis's own `DedicatedThreads` feature flag,
    `ConnectionMultiplexer.SetFeatureFlag("DedicatedThreads", true)`, set before connecting. It has nothing
    to do with OptunaSharp's executor of the same name. The flag is process-wide, its own documentation
    reserves it for support guidance, and it adds a reader and a writer thread per connection.

  **The backends set the flag by default** (plan 10, item 2.3). The URL and `ConfigurationOptions`
  constructors of the StackExchange.Redis and NRedisStack backends, and the DI `Add…JournalStorage(url, …)`
  methods, take `dedicatedThreads: true` and set the flag before they connect. `false` leaves the flag as
  it is (it never turns it off), and a multiplexer the caller passes in is never touched. Because the flag
  is process-wide, every StackExchange.Redis connection the process opens afterwards, the host's own
  included, gets its two threads too. The Redis integration page, `storage.md` and the backends' XML
  remarks say so.

  **StackExchange.Redis 2.x**, the `net8.0` builds' client (2.13.17 for the StackExchange.Redis backend,
  2.12.14 under NRedisStack 1.5.0; a process loads one), has no such flag and silently ignores the name.
  It is slow until the pool grows, and it can time out too: the measurement below passed, but on a machine
  busy with other work the Storage suite's host case (twice the pool's minimum plus 8 callers) timed out in
  1 of 3 runs on the code before plan 10's item 2.7 and 2 of 3 after it (2026-11-07). On `net8.0`, keep the
  `DedicatedThreads` executor for `nJobs` and raise the pool's minimum above a host's concurrent callers with
  `ThreadPool.SetMinThreads`; the test now pins that remedy on 2.x.

  Measured 2026-11-07 on 32 cores, pool minimum 32, each caller asking and telling 10 trials on one
  journal from `Task.Run`:

  | Callers | `net10.0`, no flag | `net10.0`, flag (the default) | `net8.0` (2.x) |
  |---|---|---|---|
  | 24 | 648 of 648 in 12.6 s | 648 of 648 in 7.3 s | 648 of 648 in 8.8 s |
  | 64 | 634 of 640 in 37.0 s; 6 failed calls (4 `RedisTimeoutException`, a `RedisConnectionException`, a `TaskCanceledException`) | 640 of 640 in 7.0 s | 640 of 640 in 37.6 s |
  | 64, pool minimum 200 | 640 of 640 in 7.6 s | — | 640 of 640 in 8.2 s |
  | 96 | — | 960 of 960 in 10.8 s | 960 of 960 in 66.6 s |
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

**The thread-pool executor** (§3.5) runs every trial on a pool thread, so an `nJobs` at or above the pool's
minimum brings these stalls back. The "Thread pool" column of §3.6 measured the plain pool this executor
queues to, and the re-measurement of the built executor there agrees. OptunaSharp leaves the choice and its
limits to the caller. When a storage or artifact client
above is in use, keep the default `DedicatedThreads` executor, or raise the pool's minimum above `nJobs`
plus the host's own need with `ThreadPool.SetMinThreads`.

The same applies without `nJobs`. A host that calls `Ask`/`Tell` from many pool threads at once starves
too: on 2026-10-29, 64 concurrent `Task.Run` callers on a Redis journal finished 624 of 640 trials in
92 s, with timeouts, and 640 of 640 in 11 s with StackExchange.Redis's `DedicatedThreads` flag. The
table above repeats the measurement on 2026-11-07, after the flag became the backends' default.

### 7.3 Throughput

- The journal holds its monitor across every round trip of an operation, as Python's `JournalStorage`
  does, so a storage's round trips are serialized. Plan 10's item 2.2 cut how many there are on a
  standalone Redis server (keys, record bytes and numbering stay Python's):
  - a sync is one call of a read-only Lua script that returns the counter and up to 1,000 records, where
    Python sends one `GET` for the counter and one per record;
  - the append script seeds the counter itself, where Python sends a `SETNX` before every append.

  A trial with two parameters, asked and told, now costs 12 client commands, every one a script call
  (8 syncs, 4 appends), against 20 before (12 `GET`, 4 `SETNX`, 4 appends), counted with
  StackExchange.Redis's profiler. Against a local Redis (0.66 ms median `PING`), a journal ran 92–127
  trials/s at `nJobs` 1 and 8, against 62–69 before, and a new storage over 400 trials started in
  24–49 ms instead of 312–391 ms. Cluster mode still reads key by key, as Python does, because the keys
  sit in different hash slots. On 2026-10-29, at a 1.1 ms round trip, the old code ran 47–49 trials/s at
  `nJobs` 1 and 8 and 23 at 16, and one `JournalStorage` per worker thread (separate monitors, one
  journal) reached 138–153 trials/s.
- **Snapshots.** A Redis journal storage saves OptunaSharp's replay snapshot under its own key,
  `{prefix}:snapshot:optunasharp`. Python's pickle stays alone in `{prefix}:snapshot`, so a new worker of
  either language restores its own latest snapshot. A snapshot is saved by the worker that creates a
  study or trial whose id is a multiple of the interval. It serializes the whole study under the storage's
  monitor, as Python saves its pickle under its lock, so it stalls that process's other operations and
  grows with the study. The backend states the interval (`IJournalSnapshotInterval`): Optuna's 100 in both
  read modes, the owner's choice on 2026-11-07 after the binary format below (a standalone server used 1000
  from item 2.2 until then). `JournalStorage(…, snapshotInterval:)` overrides it. Measured at 20,000
  two-parameter trials with one string attribute each (Release build, local Redis):

  | Interval | Writer, ms per trial | Cold start, standalone | Cold start, cluster mode |
  |---|---|---|---|
  | 100 (the default) | 9.76 | 109 ms | 0.43 s |
  | 1000 | 14.81* | 170 ms | 3.3 s |
  | none | 10.74 | 927 ms | 70.8 s |

  \* Builds ran on the machine during that phase; other work shared the machine throughout. Within noise,
  a snapshot every 100 trials costs the writer nothing: one of this study saves in about 20 ms.

  - **The snapshot format (plan 10, item 2.7).** Snapshots are OptunaSharp's own binary format, v2: each
    distribution's JSON once in a table, doubles as their eight bytes, integers as variable-length numbers,
    strings as Latin-1 or UTF-16, attributes as tagged values of PyJson's object model, decoded straight into
    trials. At 20,000 trials it is 1.6 MB, saves in about 20 ms and restores in about 39 ms, where the v1 JSON
    it replaced was 11 MB, saved in 437 ms and restored in 643 ms (same machine, same session). A restore
    gives exactly what restoring v1 gives; v1 snapshots still restore; older OptunaSharp versions cannot read
    v2 and replay the whole journal instead.
  - **Records** parse from their UTF-8 bytes through `PyJsonUtf8`, which gives `PyJson`'s values and errors
    for every input, and trials with the same distribution JSON share one distribution object (the owner's
    choice; distributions are immutable, and `FrozenTrial.DeepCopy` already shares them). Replaying 100,001
    records with no snapshot took 367 ms instead of 537 ms; a file journal's cold start 390 ms instead of 715 ms.
  - The measurements this section held before (an interval of 100 costing the writer 48%, cold starts of
    1.19–1.30 s) came from probe builds with the JIT optimizer off: `dotnet run file.cs` ignores a file's
    `#:property Configuration=Release`, and only `dotnet run -c Release file.cs` optimizes. Rerun in Release
    before the new format, the old code gave the writer 13.62, 11.10 and 11.65 ms per trial at 100, 1000 and no
    snapshot, and standalone cold starts of 962, 944 and 1143 ms.
- The file journal syncs every append to disk. 16 contending workers sustained about 80–150 appends per
  second on an SSD.
- Sampling is serial per sampler (§6.3).

## 8. Python bridges and the GIL

- `PythonRuntime` (in `OptunaSharp.Interop.pythonnet`) runs one embedded CPython per process, and every
  bridge call on **one Python thread** (plan 10, item 2.5, the owner's choice of 2026-11-07; review item
  19b). `PythonRuntime.Run(body)` hands the body to a long-lived background thread named
  `OptunaSharp Python` and waits for it. That thread starts the engine (so it holds CPython's main thread
  state for the life of the process), or, with a host-started engine, takes a thread state of its own and
  keeps it. It runs each body under the GIL, one at a time. Every access to a `PyObject`, its `Dispose`
  included, must happen under the GIL.
  - **Why.** On a .NET thread, pythonnet's `Py.GIL()` creates a CPython thread state when the outermost
    scope opens and frees it when that scope ends. pybind11 extensions remember the thread state of the
    thread on which their process-wide internals load, and use it again after it was freed.
  - **Measured on 2026-11-07.**
    - `PymooSampler` crashed with `0xC0000005` in scipy's pybind11 `cdist` (pymoo imports `scipy.spatial`
      lazily): 7 of 8 runs on a new thread, 6 of 6 on a 16 MB thread, 6 of 6 built and driven on one
      thread, 6 of 6 on a pool thread, 0 of 8 on the main thread.
    - A worker that kept one thread state passed 8 of 8, and so did importing scipy on the main thread
      first.
    - A bare probe failed 5 of 5 with scipy (a crash) and 5 of 5 with torch (a hang), and passed 5 of 5
      each with the state kept.
    - The main thread and Python's own threads keep their state, which is why Python never sees this.
  - **Why one thread.** The owner chose it over keeping a state per caller thread (which leaks a state
    whenever a .NET thread ends) and over an owner thread per sampler (which leaves the other bridges
    exposed):
    - no thread state is ever freed, and nothing leaks;
    - thread-local Python state (torch's grad mode, CUDA's current device, numpy's error state) is the
      same for every call;
    - the 16 MB stack covers the macOS numba overflow below for every bridge;
    - a call costs about 1.5 µs more: measured 1.45–1.56 µs through `PythonRuntime.Run` in a Release build,
      against 0.55–0.8 µs for the old per-call thread state and 0.4 µs for a kept one, where a bridge call
      takes milliseconds.
    - Python already runs one thread at a time, so no parallelism is lost within a study, whose sampler
      calls are serial anyway. Concurrent studies whose Python code releases the GIL (torch kernels,
      large numpy operations) no longer overlap; torch kernels use every core inside one call.
  - **Rules.**
    - A body that calls `PythonRuntime.Run` again, directly or from .NET code that Python calls back,
      runs inline.
    - A caller that holds the GIL (code inside its own `Py.GIL()`) releases it while it waits, as
      CPython's C extensions release it around blocking waits.
    - The caller's culture and `AsyncLocal` values flow into the body, and the body's exception reaches
      the caller with its stack trace.
    - A body must never wait for another thread that may itself be waiting for the Python thread: the
      body occupies the only thread that could run that call. So code inside `PythonRuntime.Run` must
      not call `Trial` or `Study` methods while other threads optimize the same study with a Python
      sampler; objectives take the GIL with `Py.GIL()` for their own Python code instead, which stays
      safe (below). OptunaSharp's bridges follow this rule: their bodies take only their own locks and
      the storages' locks.
- The bridges (`OptunaHubSampler`, `OptunaHubPruner`, `BoTorchSampler`, `PyCmaSampler`, `PymooSampler`)
  call Python inside their sampler or pruner methods, that is inside the monitor. The order is the
  monitor, then the Python thread and the GIL.
- `OptunaHubSampler` goes on under the GIL, on the Python thread: it takes its private `_gate` and reads
  the C# study through the storage to rebuild its Python shadow study. Storage calls take only storage
  locks, which come after the GIL (§5.2).
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
    when the monitor is contended and the thread holds the GIL does it release the GIL around the wait, as
    `Py_BEGIN_ALLOW_THREADS` does, and take it back once the monitor is held. The wait hook releases the
    GIL around the whole wait, as Python's own `concurrent.futures.wait` does. Without a bridge both hooks
    are absent and Core behaves as before.
  - **Which thread holds the GIL** (catalog Open item 28, fixed 2026-11-08). The hooks and the Python
    thread's hand-off (`PythonThread.Await`) ask whether the calling thread has a current CPython thread
    state: `PyThreadState_GetUnchecked` (exported from 3.13 on) or `_PyThreadState_UncheckedGet` (3.11 and
    3.12; each version exports one of the two). A thread holds the GIL exactly when it has one, and
    `PyEval_SaveThread`, the release, requires one.
    - They used to ask `PyGILState_Check`. Once any sub-interpreter has existed in the process
      (`concurrent.interpreters`, `_interpreters`, `_xxsubinterpreters`), CPython turns that check off for
      good, and it answers 1 on every thread. A thread without the GIL then released one, and the process
      ended: `Fatal Python error: PyEval_SaveThread: … the current Python thread state is NULL` on 3.12,
      an access violation in `PyEval_SaveThread` on 3.14. That happened in `Optimize(nJobs > 1)` (the wait
      for the workers), at every bridge call from a .NET thread, and at every contended monitor.
    - Measured on CPython 3.11.15, 3.12.12, 3.13.7, 3.14.2 and free-threaded 3.14.2: `PyGILState_Check`
      answers 1 on every thread after `create()` and still does after `destroy()`; the thread-state export
      answers correctly throughout. `PyGILState_Check` remains the last resort, for a build that exports
      neither name.
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
- **`PymooSampler`** works on any thread and at any `nJobs` (review item 19b, closed by the Python thread).
  Off the main thread it used to crash in pymoo's `Algorithm.ask`, and it was limited to one thread and
  `nJobs: 1`. Checked on 2026-11-07, 8 runs per mode: a new thread, a pool thread, built and driven on one
  new thread, `nJobs: 4`, and the main thread, all 8 of 8. AGE-MOEA and AGE-MOEA-2 passed 4 of 4 on a new
  thread and at `nJobs: 4`.
- **Stack size on macOS arm64.** pymoo's `AgeMoea` and `AgeMoea2` JIT-compile numba kernels on their first
  `tell`, and that needs more stack than a .NET-created thread has by default; the process stopped with
  "Stack overflow." on `nJobs` workers, `Task.Run` work and test runners. CPython gives its own threads 16 MB
  on macOS (`THREAD_STACK_SIZE`).
  - Bridge calls now run on the Python thread, which has 16 MB, so `PymooSampler` needs no remedy.
  - The `nJobs` dedicated threads reserve 16 MB too (the owner's choice of 2026-11-07), for objectives that
    run such Python code themselves.
  - Python code you run on threads of your own, the thread pool included, still needs one of the remedies
    on the pymoo page: a thread with a `maxStackSize` of at least 16 MB, or `DOTNET_DefaultStackSize=1000000`
    (hexadecimal) set before the process starts.
  - Windows x64 and Linux x64 are not affected.
  - The stack is reserved, not committed (Windows `STACK_SIZE_PARAM_IS_A_RESERVATION`; Linux and macOS
    map it lazily), so an unused reserve costs address space only.

## 9. Numeric kernels and random generators

- **NumSharp's kernels** are single-threaded by default (`np.multithreading` is off), which also keeps
  their summation order fixed.
  - `NUMSHARP_MULTITHREADING=1`, or `np.multithreading(true)`, lets NumSharp split large kernels over
    thread-pool threads (`Parallel.For`, up to `NUMSHARP_NUM_THREADS`, 8 by default). In NumSharp
    0.80.0, as in 0.70.0, that covers only the large 1-D floating-point dot product.
  - The split changes the summation order, so the last bits of a result change and a run no longer
    matches Optuna. The split work also competes with the host's work for the pool (§7.2).
  - OptunaSharp never turns it on.
- **The linear algebra** (CMA-ES's eigendecomposition, the GP stack's Cholesky, NSGA-III's hyperplane
  solve) goes through `NumSharp.Interop.OpenBLAS`.
  - With `nJobs: 1` OptunaSharp sets no OpenBLAS thread count, so the library's default applies, as it does
    in NumPy. That is what keeps a sequential run byte-identical to Python.
  - The low bits of a result depend on that count (NumSharp measured four different answers at 1, 2, 4
    and 24 threads) and on the CPU kernel OpenBLAS dispatches.
  - To pin the count for the whole process, set `OPENBLAS_NUM_THREADS` in the shell before the process
    starts, or call `OpenBlasEngine.Enable(threads: n)`. `Environment.SetEnvironmentVariable` inside the
    process does not reach the native library.
  - A byte-parity run against Python needs the same count there, set before `import numpy`.
- **The OpenBLAS thread policy** (`TrialExecutorOptions.BlasThreads`, [plan 11](plans/11-threading-round-2.md),
  2026-11-08). OpenBLAS's worker threads are one pool for the process, sized for one caller, and `nJobs`
  workers in linear algebra at once oversubscribe the cores.
  - **`null`, the default, is the automatic policy:** OpenBLAS untouched with `nJobs: 1`, single-threaded while
    an `nJobs` call with several workers runs. A positive count is held for the duration of any call,
    `nJobs: 1` included.
  - Measured 2026-11-08 on 32 cores (OpenBLAS starts with 24 threads), 2,000 trials of a 40-dimensional CMA-ES:
    4.50 s at `nJobs: 1`; at `nJobs: 8`, 5.22 s with OpenBLAS's 24 threads and 3.58 s single-threaded. A GP
    study showed no difference (18.66 s against 18.68 s): its matrices are too small for OpenBLAS to thread.
  - `Internal/BlasThreadPolicy` keeps one list of the counts held by running calls. The count in effect is the
    smallest, so concurrent calls never oversubscribe; the count before the first holder is restored when the
    last one releases, on every path out of `Optimize`, the objective's exceptions included. The native setting
    is touched only on the first hold, a change of the smallest count and the last release, so calls that agree
    never call OpenBLAS while another call's workers are inside it.
  - It changes the count through NumSharp's threading knob, which also writes `OPENBLAS_NUM_THREADS` into the
    process environment and the C runtime's copy; the policy removes the process variable again on restore
    when it was not set before.
  - The count is process-wide: while a call holds it, every OpenBLAS call in the process runs with it, the
    host's own included, and an `nJobs: 1` call on the automatic policy runs with whatever another call holds.
  - It saves no memory: OpenBLAS commits memory for its threads when it loads (788 MB of private bytes with 24
    threads against 50 MB with one, measured; the working set stays near 44 MB), and only
    `OPENBLAS_NUM_THREADS=1` set before the process starts avoids that.
  - Without a loaded OpenBLAS the policy does nothing, and no failure in it fails an optimization.
- **Concurrency.** The sampler monitor admits one call per sampler at a time. Samplers of different
  studies, and per-worker instances (§6.3), can run linear algebra at the same time; OpenBLAS's own worker
  threads are one pool for the process, which is what the policy above manages.
- **Random generators.** Since NumSharp 0.80.0 each draw holds the bit generator's lock, as numpy's
  does, so concurrent draws no longer corrupt a generator. Before 0.80.0 they did (§6.3).
  - Each of OptunaSharp's generators belongs to one object. A sampler draws its generators under its
    monitor. The others live for one computation (a forest fit, a cross-validation split), except the
    terminator evaluators', which outlive their calls (§6.6).
  - NumSharp's `np.random` is one generator for the whole process, locked per draw like Python's global
    `np.random`. An objective that draws from it under `nJobs` is safe, but every worker contends for
    the one lock, and the workers' draws interleave in an order no run repeats.

## 10. Writing thread-safe code against OptunaSharp

- **Objectives.** Under `nJobs` the objective runs on several threads at once, so the state it shares
  must be thread-safe. Each trial gets its own `Trial` object. Draw random numbers from a generator per
  thread or per trial (`Random.Shared`, or a generator seeded from `trial.Number`), never from one shared
  `System.Random`. NumSharp's `np.random` is safe to share since 0.80.0, but every worker contends for its
  lock (§9).
- **Callbacks** run on the worker thread, concurrently with other trials' callbacks. They may call
  `Stop()`. `TerminatorCallback` serializes its own calls (§6.6).
- **Custom samplers and pruners.**
  - The study already serializes its calls into one instance under the instance's monitor; do not add
    a lock for them. Calls made outside a study (a test, a tool) are not serialized.
  - Avoid static mutable state, and keep per-study state keyed by the storage object and the study id:
    study ids repeat across in-memory storages.
  - Do not tell trials of the served study from a hook (§5.3).
  - A Python-backed implementation runs its Python inside each method through `PythonRuntime.Run`,
    which runs it on the Python thread (§8). Calls into OptunaSharp from code that holds the GIL with
    `Py.GIL()` are safe (§8). What stays unsafe: a lock of its own that it holds while it calls
    `PythonRuntime.Run`, if a body could wait for that lock; and a body that waits for another thread
    which may be waiting for the Python thread.
- **Custom storages.** Every operation must be safe for concurrent calls on one instance: the workers,
  the heartbeat thread and readers all call it. `TellLockFor` comes from `BaseStorage`. Make `Dispose`
  idempotent.
- **Log handlers** receive one record at a time per handler (§6.6), so they need a lock only for state
  they share with other code. They must not throw, must not call OptunaSharp or take the GIL, and should be
  quick: other threads that log wait for them (§5.3).
- **Choosing `SamplerConcurrency`** (§6.3). Keep `Shared`, Optuna's model, unless sampling limits your
  `nJobs` run (a multivariate TPE over many parameters, a high-dimensional CMA-ES, a fast objective). Then
  `PerWorker` lets the samplers that take part propose in parallel; the others stay shared on their own. Make
  the functions you hand to such a sampler thread-safe.
- **Choosing `BlasThreads`** (§9). Keep the automatic policy (`null`) unless the host runs its own OpenBLAS
  work while a study optimizes and needs more threads for it, or you want one count everywhere (set it
  before the process starts instead).
- **Trial records.** `GetTrials(deepcopy: false)` returns the storage's own records, shared by every
  thread. Read them, never mutate them, as Optuna documents for `deepcopy=False`. `Trials` and
  `GetTrials()` return copies.
- **Many workers creating one study.** `CreateStudy(loadIfExists: true)` from many workers at once is
  safe. The storage lets one worker create the study, and the others get `DuplicatedStudyError` and load
  it, as in Python.
- **`OptunaSearchCV(nJobs: n)`** passes `n` to `Optimize`, so `n` cross-validations run at once.
  - The native sklearn port clones its estimator for each, which is safe.
  - `MLNetEstimator` shares one caller-supplied `MLContext`, its fold views and your pipeline factory
    across every call. Tested at `nJobs: 4` with a FastTree regression pipeline: every trial's fold scores
    equal a sequential evaluation of the same parameters (plan 10, item 2.6). Your factory and scorer run
    concurrently, so they must be thread-safe: build a new trainer per call, and never share a
    `PredictionEngine`, which ML.NET documents as not thread-safe. A trainer that draws randomness from the
    shared context (bagging, a feature fraction below 1, shuffling) was not tested under `nJobs`, and its
    results can depend on the order the trials run in.
  - `MLNetEstimator` records an exception raised while a fold fits or scores as that fold's error score
    (NaN by default), as scikit-learn does. A thread-safety fault in your pipeline therefore shows up as
    NaN or changed scores rather than as an exception.
- **Artifacts.** Uploads from concurrent trials do not interfere. Each upload gets a new artifact id (a
  GUID) and its own file or object, and one attribute write records it on the trial or study.
- **Choosing an executor** (§3.5). Keep the default, `DedicatedThreads`, unless you have a reason to put
  trials on the thread pool, such as a host that sizes and monitors its pool itself. Set the choice once
  for the process (`TrialExecutorOptions.Default`), per DI container
  (`OptunaStudyOptions.DefaultExecutorOptions`), per study (`study.ExecutorOptions`), or per call
  (`executorOptions:`). With the pool, keep `nJobs` below the pool's minimum, or raise the minimum, when
  the storage client needs the pool (§7.2). A derived `TrialExecutorOptions` may compute its values; each
  call reads them once.
- **Host applications.** Do not run many studies or Ask/Tell calls on pool threads at once when the
  storage client needs the pool (§7.2). Use `nJobs` or threads of your own, or raise
  `ThreadPool.SetMinThreads`. For Redis, the StackExchange.Redis and NRedisStack backends turn on
  StackExchange.Redis's process-wide `DedicatedThreads` flag by default; keep that default
  (`dedicatedThreads: true`) unless the host manages the flag itself.
- **Dependency injection.** Samplers and pruners are registered transient by default, so every study
  gets its own instance. Storages are singletons by default: one instance is thread-safe and shares its
  connection.

## 11. Tests that pin the design

| Test class | Pins |
|---|---|
| `OptimizeConcurrencyTests` | `nJobs` draws never repeat (Random, TPE); samplers with their own state complete every trial; evolution strategies never repeat a candidate; Hyperband builds its brackets once; `Optimize` on pool threads after an `nJobs` run is not a nested call; the error rules of §3.3; workers on dedicated threads; a trial's writes leave older snapshots alone; concurrent user attributes; the `nJobs` refusal |
| `OptimizeEdgeCaseTests` | the stop, timeout and error rules of §3.3–3.4, each first observed on Optuna 5.0: timeouts; `Stop()` from an objective, a callback or another thread; `MaxTrialsCallback`; `catch`; failing callbacks and sampler hooks; unacceptable results; a nested `Optimize` in a worker; a failing final write and a trial failed as stale under its objective (both `[Misaligned]`); the timeout-only progress bar |
| `OptimizeCancellationTests` | plan 10, item 2.4: an already cancelled token runs no trial; a cancellation mid-run finishes the trials in flight and then throws, carrying the token (`nJobs` 1 and 8, both executors); an objective observing the token is stored FAIL whatever `catch` holds, and the callbacks skip it; workers observing it leave no trial RUNNING; an objective ignoring it finishes COMPLETE; a cancellation exception of another token is an ordinary failure; a linked token is reported with the caller's; `Trial.CancellationToken` in `Optimize` and for asked trials; `Stop()` unchanged; a cancelled run on a journal leaves no trial RUNNING |
| `StopSignalsTests` | the first SIGINT or SIGTERM is held and cancels the token, a second falls through; after `Dispose` signals fall through and the token stays readable; a throwing token callback does not escape the handler; a signal during `Optimize(nJobs: 4)` stops it gracefully; a real SIGTERM sent to the process cancels the token (Linux and macOS) |
| `TrialExecutorOptionsTests` | the executor of §3.5: new options are dedicated threads with a 30 s idle timeout; both settings and `Default` refuse invalid values; `ThreadPool` runs every trial on a pool thread and `DedicatedThreads` none; `nJobs: 1` runs on the caller's thread whatever the executor; `nJobs` trials at once on the pool below its minimum; an `Optimize` called from a pool thread never runs a trial on that thread; pool threads keep no loop state afterwards; Python's error and stop rules on both executors; the caller's culture and `AsyncLocal` values reach workers and callbacks on both; the order of the four levels; a new `Default` reaching a study that never chose; `CreateStudy` and `LoadStudy` leaving the study setting unset; the snapshot (a change during a run does not reach it; a derived class's four getters are read once, and its invalid values are refused before any trial); equal idle timeouts sharing one thread set; a short timeout retiring its threads, which then restart; the DI factory giving every study its options; new options share the sampler and leave OpenBLAS to the automatic policy; `SamplerConcurrency` and `BlasThreads` refuse invalid values |
| `SamplerConcurrencyModeTests` | §6.3's `SamplerConcurrency`: by default every call reaches the study's sampler one at a time and no worker instance is made; `PerWorker` has four workers sampling at the same instant, one instance each, the study's own serving none; each trial is set up, sampled and told by one instance, reseeded before it; a sampler without worker instances stays shared and is asked once; `nJobs: 1` ignores the setting; no reused thread (dedicated or pool) keeps a worker's instance; the pool reuses instances per sampler and never hands an old sampler's out for a new one; exactly the listed samplers make instances (reflection), exact copies only; each taking-part sampler at `nJobs: 8` never proposes a point twice |
| `BlasThreadPolicyTests` | §9's OpenBLAS policy, from a known count of 3: untouched at `nJobs: 1`; single-threaded at `nJobs: 4` on both executors, then restored with the variable removed; an explicit count at any `nJobs`; a variable set before the call kept; the count restored when the objective throws; two concurrent calls running with the smallest count, the survivor getting its own back, the last restoring; holds nesting by count, a null hold doing nothing |
| `LoggingTests` | §6.6: one write and one flush per record, text unchanged; eight threads logging failures through one `StreamLogHandler` over an unsynchronized writer keep every record whole; a handler added twice receives each record once, as Python |
| `StudyConcurrencyEdgeTests` | two threads optimizing one study; many threads asking and telling; a consistent reader during a run; enqueued trials run once and first, also when callbacks enqueue them; two tells of one trial; one sampler shared by two studies on two threads |
| `JournalConcurrencyTests` | both lock protocols under contention; stale-lock takeover after the grace period; mixed protocols on one journal; separate storages on one study; a shared backend's offset cache; many studies on one journal; enqueued trials popped once; one trial's concurrent reports, suggestions and constraints; the create race |
| `HeartbeatTests` | heartbeat threads beat and stop; a failed write ends only its thread, reported like `threading.excepthook`; trials complete when every heartbeat write fails; `FailStaleTrials` and the retry callback; `Ask`'s warning outside `Optimize` |
| `SamplerNJobsSweepTests`, `PrunerNJobsSweepTests` | every Core sampler and pruner under `nJobs` on a journal: contiguous numbers, every trial ends as its rule dictates |
| `SamplerParallelReadTests` | §6.3, plan 12: every Core sampler, read by eight trials at once on eight threads and in lockstep, hands no point to two of them, and the samplers that never re-propose a finished point repeat none (CMA-ES and VkD-CMA one, through Optuna's seeding); per-worker instances; populations larger than a round; whole `Optimize(nJobs: 16)` runs evaluating no point twice; Grid and BruteForce (both modes, `[Misaligned]`) over 88 of 90 points; DE and PSO with more trials than members, the rest sampled independently without warnings; the sequential repeats of the samplers allowed them |
| `BridgeSamplerParallelReadTests` (interop) | §6.3, plan 12: each Python bridge's repeats under parallel reads, exact in lockstep (PyCma's compared bit for bit with `optuna_integration.PyCmaSampler`) and bounded on threads; BoTorch in the slow category |
| `LazyRandomStateSeedingTests` | unseeded samplers created together never share a stream; seeded ones stay reproducible |
| `OsEntropySeedTests` | numpy's OS-entropy seeding: the constructor's key, position 623 and draws, and a reseed's kept position and dropped Gaussian, against numpy 2.4.2 output for injected entropy; no repeated draw in 300,000 reseeds |
| `TerminatorTests` | `TerminatorCallback` at `nJobs: 8` runs its terminator one call at a time |
| `ProgressBarTests` | the bar counts every `nJobs` trial once, in order |
| `ConcurrentOrderedDictTests` | the contract of the internal concurrent collection (§5.1) |
| `ConcurrencyGunTests` | the harness below: shooters are dedicated threads, halted at the gate and released into the shot together; the callbacks run while every shooter is halted; failures are reported by round and thread. It catches a lost update in an unguarded read-modify-write, with a control that must show nothing, and pins that NumSharp's generator (0.80.0 and later) never tears when drawn without the sampler monitor (§6.3). A lock-order deadlock is reported as a hang, and its shooters are freed |
| `TrialConcurrencyCoverageTests` | one trial driven by every shooter at once (§6.2, §6.4, §6.5): one report of a step stored, and the trial's view agrees with the storage; one name sampled once; distinct names each sampled once with distinct draws; several tells (through the `Trial` and through its number) store one result, refuse the rest, and run `AfterTrial` once; `skipIfFinished` tells all return the stored result; pruning decisions made on each trial's own values, one at a time |
| `StudyConcurrencyCoverageTests` | one study used by every shooter at once (§6.1, §7.1), in memory, through one journal storage, and through one journal storage per thread: asks get distinct trials numbered without a gap; an enqueued trial goes to exactly one asker; a waiting trial claimed through the storage has one winner; `CreateStudy(loadIfExists: true)` creates the study once |
| `InfrastructureConcurrencyCoverageTests` | the sampler, pruner, tell-stripe and `MonitorGate` monitors admit one thread at a time (§5); a reseed never tears a generator others draw from; one sampler serves a study per thread without repeating a draw (§1.5); a categorical index built on first use maps every choice; the worker scheduler runs every task queued at once exactly once, all in flight together (§3.5); log handlers changed while others log never cost a record (§6.6); each handler serves one record at a time while different handlers serve different threads at once (§6.6); artifact uploads to one trial at once each get their own id, file and record (§10); `TerminatorCallback` runs its terminator one call at a time (§6.6); the warning hook delivers every warning while subscribers come and go; readers of trials, best trial, trials table and summaries during writes always see a consistent, growing study |
| `StorageConcurrencyCoverageTests` | in memory and on a journal (§6.1, §7.1): a record read without a deep copy never changes under its reader while others write it; every write of concurrent writers lands, and a fresh journal replay sees what the live storage holds; the best trial stays the best of all concurrent tells, minimized and maximized; study user attributes set at once are all kept |
| `OptimizeConcurrencyCoverageTests` | one `Study` object (§3.4, §4, §6.1): a sampler or pruner swapped without pause while others work runs every call on one instance; concurrent `Optimize` calls are never nested invocations, and `Stop()` outside a loop is refused; `Optimize(nJobs: 2)` from every thread at once gives one perfect study; one callback's `Stop()` stops every `Optimize` call on the object |
| `SamplerSharingConcurrencyCoverageTests` | one sampler instance serving a study per thread at once, every study with id 0 (§1.5, §6.3): `PerStudyState` resolved for every study at once keeps one state per study; a genetic sampler gives each study the generations of a solo run; each of 29 Core samplers completes every trial of every study |
| `MLNetEstimatorNJobsTests` (Integration.ML.NET suite) | `OptunaSearchCV(nJobs: 4)` over one `MLNetEstimator`: 16 trials that overlap in time, every fold score equal to a sequential evaluation and none NaN |
| `StorageConcurrencyTestsBase` and its SQLite, PostgreSQL, MySQL and gRPC subclasses (Storage suite) | one storage under `nJobs: 8`, eight separate storages, enqueued pops, sixteen storages creating one study name |
| `RedisJournalConcurrencyTests` (Storage suite) | separate storages on one study; `nJobs` at four times the pool's minimum does not starve the client; dispose races |
| `RedisJournalCostAndThreadsTests` (Storage suite) | plan 10, items 2.2 and 2.3: a two-parameter trial costs 12 client commands, all script calls; snapshots at the configured interval under OptunaSharp's key only; with no snapshot of its own a backend reads only the first byte of Python's pickle; the owning constructors turn the `DedicatedThreads` flag on by default, `false` leaves it alone, an injected multiplexer is untouched (Inconclusive on 2.x, which has no flag); more thread-pool callers than twice the pool's minimum ask and tell without losing a trial, on 3.x and on 2.x |
| `JournalBackendContractTests` (Storage suite, all three clients) | batched reads across batches, ending at the counter seen first and waiting for a record missing inside a batch; the counter seeded inside the append script after the journal's keys are deleted; snapshots written to OptunaSharp's key only, an older one found once in Python's key, a pickle there ignored; each backend's snapshot interval (Optuna's 100 in both read modes), taken by a storage built without one |
| `JournalStorageTests` | the snapshot interval: saves at its multiples, a cold start that reads only the records after the snapshot, the backend's interval when the constructor is not given one (read once), Optuna's 100 otherwise, and invalid values refused from either source |
| `PyJsonUtf8Tests` | plan 10, item 2.7: the UTF-8 record reader against `PyJson`, strictly (runtime types, key order, double bits, exception type and message): every literal and number form, lenient and refused ones included; every escape; raw UTF-8 text, valid and invalid; non-ASCII bytes outside strings; malformed structure; repeated keys; the depth limit; journal records of every operation in three separator styles; 18,000 random documents; 4,500 corrupted ones; decoded text; eight threads at once; the state after a failed or huge parse. Valid documents must take the fast path, and refused ones must not |
| `JournalSnapshotV2Tests` | plan 10, item 2.7: for random journals of every record kind, restoring the binary v2 snapshot gives the original state and what restoring v1 gives (v1's bytes, then every field strictly), and the v2 bytes survive a round trip; empty states; deleted studies keeping their enumeration order; shared distribution objects, never the deprecated ones, and the lookup's capacity; v1 snapshots restoring; every truncation and 6,000 corruptions restoring nothing without throwing; values a record cannot carry taking v1's normalization; v1's exception kept; attributes at the depth limit; a storage restoring v2 seeing what a full replay sees |
| `JournalCodecParityInteropTests` (Interop suite) | plan 10, item 2.7: journals live Optuna writes with every value kind, every trial state and every distribution type parse through the fast path to `PyJson`'s values (file and Redis forms), and replay, directly or through a v1 or v2 snapshot, into the studies Python sees |
| `JournalRedisInteropTests` (Interop suite) | each client against live Optuna on one Redis journal, both directions; Python's and OptunaSharp's snapshots on one journal, each in its own key, each language's cold start restoring its own without a failed-restore warning |
| `StorageDisposalEdgeTests` (Storage suite) | a scoped SQLite storage, gRPC proxy or EF Core storage, disposed through both of its DI registrations, synchronously or asynchronously, closes its connection, channel or owned data source |
| `JournalMixedFleetInteropTests` (Interop suite) | C# and Python workers on one journal file (both lock protocols), on Redis, and with CMA-ES, compared with a Python-only fleet |
| `CmaEsMixedFleetInteropTests` (Interop suite) | C# and Python workers taking turns on one CMA-ES study, with and without margin, run the study Python runs alone |
| `GilAwareMonitorTests` (Interop suite) | P3-1: `PythonRuntime.Initialize` installs the gate's hooks; three deterministic two-thread deadlocks (a GIL holder against a sampler, trial or pruner monitor held by a thread that needs the GIL), recovered by `Thread.Interrupt` if they regress; an objective inside `Py.GIL()` at `nJobs: 4` over a GIL-taking sampler and pruner, on dedicated threads and on the thread-pool executor; `Optimize(nJobs: 4)` called inside `Py.GIL()`; four ask-and-tell threads holding the GIL. Catalog Open item 28: after a sub-interpreter, threads without the GIL never release it, at the `nJobs` wait, a contended sampler monitor and a bridge call from the main thread, plus a control without one; each in a child process (`test/OptunaSharp.Tests.Interop.ChildProcess`), because a sub-interpreter changes CPython for the rest of its process and a regression ends it |
| `PythonThreadTests` (Interop suite) | plan 10, item 2.5: calls from six .NET threads run on one Python thread whose `threading.local` keeps its value between them; a nested `Run` runs inline; a caller holding the GIL completes; the caller's culture and `AsyncLocal` flow into the body; the body's exception reaches the caller unchanged; the Python thread reserves 16 MB (Windows); `PymooSampler` off its thread and at `nJobs: 4` completes every trial (review item 19b); torch imported from one thread and used from others does not hang; the P3-1 deadlock shape through `Run` completes |
| `OptunaHubBridgeParityTests` (Interop suite) | the real hub sampler at `nJobs: 8`; the real hub sampler and median pruner with an objective inside `Py.GIL()` at `nJobs: 4` |

The tests plan 10's item 2.1 added were checked against the old code, which they fail: 9–14 repeated
draws in 300,000 reseeds, and 8 terminator calls at once. For items 2.2 and 2.3, the old code costs 20
client commands per trial where the test expects 12, writes its snapshot to Python's key, and, without
the flag on 3.x, lost 6 of 640 trials in the host scenario. Python logs ``Failed to restore
`JournalStorageReplayResult`.`` when OptunaSharp's JSON sits in its key, which is what the cross-language
snapshot test watches for. Item 2.4's cancellation tests fail when the loop drops its token check (4 cases)
or lets `catch` swallow the run's cancellation (2 cases). On the old `PythonRuntime`, item 2.5's thread-state
test failed (the calls ran on different Python threads, and the `threading.local` lost its value), and the
stack read 1.5 MB instead of 16. Item 2.7's parity suites caught four deliberate bugs: a restore that kept the
UTC instant instead of converting it to local time (10 cases failed), a repeated key keeping its first value (1),
parameters written in their external representation (9), and studies written in id order (9). On the old
`GilAwareMonitor`, the Open item 28 test passed its control and failed its three sub-interpreter modes, each with
`Fatal Python error: PyEval_SaveThread` (CPython 3.12.12, exit 0xC0000409); the same build crashed in all three on
3.14.2 and free-threaded 3.14.2 too.

**The `ConcurrencyGun`** (`test/OptunaSharp.Tests/Concurrency/`, 2026-11-08) is the harness behind the last seven
unit-test classes in the table:
- It allocates its own shooter threads (8 by default) and halts each on the round's `ManualResetEventSlim` gate with
  a blocking `Wait`. Once all are parked it opens the gate, and the woken shooters spin in a lineup until all are
  awake, so the shots of a round start about 1 µs apart (median, 32 logical CPUs).
- `beforeRound` and `afterRound` run on the test's thread while every shooter is halted, so a test sets up and checks
  each round without locks.
- A round over its timeout is reported as a hang. The shooters still inside the shot are interrupted, and the report
  names them.
- A test that expects a race calls `ConcurrencyGun.RequireParallelHardware()` first (inconclusive on one CPU) and has
  a control: the same volley without the race, which must show nothing.

Each guard the coverage classes pin was removed in turn from a scratch copy of Core, in two waves, and 15 of 16 such
mutations failed the tests that pin them:
- the trial monitor around `Report` (one step written several times), around `Suggest` (one name sampled several
  times) and around `SetUserAttr` (the trial's view and the storage disagree);
- the sampler monitor (4 tests failed), the tell stripe (3) and the terminator callback's lock;
- a second read of `study.Sampler` inside the sampler monitor, which ran a call on the sampler it had not locked;
- the scheduler starting one thread too few;
- the in-memory storage letting a claim take a trial that is not waiting, writing a user-attribute dictionary in
  place, and setting study attributes without its lock (211 of 1,600 keys kept);
- `Trial.UserAttrs` handing out the live dictionary;
- the optimize-loop flag shared by every thread;
- `PerStudyState` keyed by study id alone, which only the direct test caught: a genetic sampler's trial anchor
  rebuilds a state found under another study's key, so its generations stay right;
- `Emit` iterating the live handler list.

Two tests passed against their first mutation and were sharpened: the report test now counts storage writes, and the
per-study keying got its own test. Publishing the categorical index before filling it passed. A missed lookup falls back to the linear scan, which still
maps every choice correctly, so the bug costs only the fast path.

The tests of plan 11 (2026-11-08) were checked the same way, with 18 mutations in three scratch builds, and every one
of them failed at least one test it should:
- the handler lock removed: eight threads inside one handler, and 1,595 of 16,000 log lines lost over a `StringWriter`
  (`Destination is too short` inside the writer); one lock for every handler instead: the per-handler test timed out;
  a record written in two halves, and the duplicate check in `AddHandler`: their tests failed;
- the automatic OpenBLAS policy gone, `Max` for `Min`, the variable never removed, and a null policy that always pins
  (`BlasThreads ?? 1`): the policy tests failed (the trials saw 3 threads instead of 1, 2 instead of 1, a stale
  variable, 1 at `nJobs: 1`); the count not validated: the derived-options test failed, and the 0 it let through was
  held for ever, since an empty lease is 0, so `Hold` now ignores counts below 1;
- the worker pool never created: the four first sampling calls missed their rendezvous; the tell reaching the study's
  sampler, the override left on the thread, a pool for `nJobs: 1`, the pool ignoring which sampler an instance belongs
  to, `GridSampler`'s instance removed, a `RandomSampler` copy made for subclasses with a fixed seed while the worker
  reseeded the study's sampler instead of its own, and `PerWorker` as the default: their tests failed, the last but
  one with 40 repeated points for `RandomSampler` and 7 each for CMA-ES and VkD-CMA (whose copied independent
  samplers then shared a stream).

## 12. Known limits and open items

[Plan 10](plans/10-threading.md) (decided 2026-11-07) schedules the work on these items. Its Phase 1, the
thread-pool executor option, was built on 2026-11-07 (§3.5). Phase 2 covers the items marked "plan 10, 2.x".

- **Journal throughput:** one monitor across all of an operation's round trips, as in Python (§7.3).
  On a standalone Redis server a two-parameter trial costs 12 round trips (plan 10, 2.2, done); cluster
  mode still reads key by key.
- **Snapshot cost:** a snapshot is still saved under the storage's monitor, as Python saves its pickle
  under its lock, but the binary format takes about 20 ms at 20,000 trials (§7.3; plan 10, 2.7, done).
  Older OptunaSharp versions on the same Redis prefix cannot read it and replay the whole journal.
- **The Redis cluster append** (§7.1): a writer that dies between `INCR` and `SET` stalls every reader,
  as in Python.
- **Ctrl+C without a handler** still ends the process and leaves the trials in flight `RUNNING` (§3.4),
  as .NET does by default. `StopSignals` or a host's `ApplicationStopping` token stops gracefully; both
  are opt-in (plan 10, 2.4, done).
- **No journal heartbeat:** a trial whose process dies stays `RUNNING`, as in Python.
- **One Python thread** (§8): bridge calls of concurrent studies run one at a time, so Python code that
  releases the GIL (torch kernels, large numpy operations) no longer overlaps across studies. A
  `PythonRuntime.Run` body must not wait for a thread that may be waiting for the Python thread.
- **Repeated candidates** (§6.3): no Core sampler hands one point to two trials asked at once in one process, but
  the genetic samplers, DE, JANUS-DE, PSO and LM-CMA re-propose finished points by design, as in Optuna; two
  processes on one storage are steered apart only by the stamped particles, targets and design rows; Grid at the end
  of its grid and the Python bridges keep their Python behavior. Plan 12 lists the follow-ups.
- **Serial sampling for the samplers that do not take part in `PerWorker`** (§6.3): GP, SAASBO and the other
  slow model-based samplers still propose one at a time per study, so at `nJobs: 8` a GP study ran slower than at
  `nJobs: 1` (18.7 s against 12.1 s, 120 trials of a 20 ms objective). Optuna has the same cap.
- **No async API**: `Optimize`, the storages and the samplers are synchronous, as in Optuna; an async objective
  blocks its worker thread. The scope of a full async stack was evaluated on 2026-11-08 and not taken up
  ([plan 11](plans/11-threading-round-2.md)). A measured evaluation of the narrower pieces (an `OptimizeAsync` on a
  bounded scheduler, pipelined Redis cluster reads, an O(1) submit loop) is catalog Open item 30, with the owner's
  decisions pending.

## 13. Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|---|---|
| The GIL serializes Python code | Monitors: sampler, pruner, trial, tell stripes, storages |
| A thread blocked on a lock or in `wait()` has released the GIL | `MonitorGate`: with a Python bridge loaded, a thread holding the GIL releases it while it waits for a tell stripe, trial, sampler or pruner monitor, or for its `nJobs` workers (P3-1, fixed 2026-11-06). Whether a thread holds the GIL is read from its own CPython thread state, which no sub-interpreter changes (Open item 28, fixed 2026-11-08) |
| numpy's `RandomState` locks each draw | NumSharp's `RandomState` locks each draw too, since 0.80.0; the sampler monitor keeps each sampler call (many draws plus state updates) whole |
| `ThreadPoolExecutor(max_workers=n_jobs)`, threads reused within the call | By default OptunaSharp's own dedicated threads, reused across trials and calls (P3-4, fixed 2026-11-06); on request the .NET thread pool (§3.5) |
| The executor is fixed | `TrialExecutorOptions`, chosen per call, per study, per DI container or for the process, and read once when the call starts (§3.5) |
| `wait(FIRST_COMPLETED)`, then `f.result()` | `Task.WhenAny`, then the first failure's original exception |
| The `with ThreadPoolExecutor(...)` exit | `WaitForAllWithoutRaising` |
| Executor threads start without the caller's `contextvars`; the locale is process-wide | Workers inherit the caller's `ExecutionContext`: its culture and `AsyncLocal` values |
| The interpreter waits for executor threads before it exits | Workers are background threads: `Optimize` waits for them, a process that ends does not |
| `KeyboardInterrupt` in the objective fails the running trial (`n_jobs=1`), or the running trials finish (more workers); then it propagates | Cancelling `Optimize`'s `ct` is the counterpart (§3.4): no trial starts, the trials in flight finish (an objective that observes the token is stored `FAIL`), then `OperationCanceledException`. Ctrl+C reaches the token only through the opt-in `StopSignals` or a host's `ApplicationStopping`; otherwise it ends the process, and the trials in flight stay `RUNNING` |
| `study._thread_local` | `Study.ThreadLocal` (`ThreadLocal<ThreadLocalStudyAttribute>`) |
| `worker_id = prefix + threading.get_ident()` | `prefix + Environment.CurrentManagedThreadId` |
| Heartbeat on a `threading.Thread`; `threading.excepthook` reports its failure | A background `Thread`; the same report, written to standard error |
| `RandomState()` and `RandomState.seed()` take 128 bits of OS entropy for the whole key | The same construction, ported: a `SeedSequence` expanded into the key, numpy's starting position, the position kept by a reseed (plan 10, 2.1; §6.3) |
| Logging handlers take their own lock per record (`Handler.handle`), and `StreamHandler` writes a record in one write | The same since 2026-11-08: a lock per handler instance, one write and one flush per record (§6.6) |
| `Logger.addHandler` ignores a handler already registered | The same since 2026-11-08 |
| `n_jobs` threads share the study's one sampler; samplers sample in parallel only as separate processes | `SamplerConcurrency.Shared`, the default, is the same; `PerWorker` gives each worker an instance of its own, as separate processes would have, for the samplers that propose no point twice that way (§6.3) |
| GPSampler limits its own OpenBLAS and torch threads to one (`optuna/_gp/thread_limiting.py`) | `BlasThreads`: OpenBLAS single-threaded while several `nJobs` workers run, whatever the sampler; untouched at `nJobs: 1` (§9) |
| Two tells of one trial: the GIL makes one practically always win | The tell stripe guarantees it |
