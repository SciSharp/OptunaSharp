# Plan 10 — Threading: a configurable `nJobs` executor, then six follow-ups

> **Status: Phase 1 BUILT (2026-11-07, §1.9); Phase 2 not started.** Phase 1, the executor, came first.
> Phase 2 starts only after Phase 1 is finished; the owner's words (2026-11-07): "document what we will do
> but we only tackle these AFTER finished all the threading". The decisions below are the owner's answers
> to three rounds of design questions on 2026-11-07. The names, deferred at first, were settled on
> 2026-11-07 before the commit (§1.8). The design document,
> [`docs/THREADING.md`](../THREADING.md), describes what the code does: §3.5 the executor, §12 the open
> items.

Context: `9cfcb12` (2026-11-06, review item P3-4) moved `Optimize(nJobs)` onto OptunaSharp's own reused
dedicated threads (`Internal/TrialWorkerScheduler.cs`). The owner had asked on 2026-10-29 for "reused
threads and the thread pool supported"; the pool option was left unbuilt because its API is the owner's
call. The measurements behind every choice are in THREADING.md §3.6 and §7.2 (2026-10-29, 32 logical
CPUs, pool minimum 32).

---

## Phase 1 — the `nJobs` executor

### 1.1 Goal

Let a caller run `Optimize(nJobs)` trials on the .NET thread pool instead of OptunaSharp's reused
dedicated threads, without changing the default or Python's semantics.

### 1.2 Decisions (owner, 2026-11-07)

| Question | Decision |
|---|---|
| Default executor | OptunaSharp's reused dedicated threads, as today |
| How a caller chooses | four levels: a process-wide static default, a DI default, a study setting, a per-call argument; each later level overrides the earlier ones |
| When it is resolved | at the start of each `Optimize` call; the first level that is set wins; the study setting is `null` until someone sets it |
| The type | an **options class** (`TrialExecutorOptions`): mutable, **not sealed** (inheritable), with a public parameterless constructor; `Optimize` takes a snapshot of it when it starts |
| Its settings | the executor kind, and `ThreadIdleTimeout` (dedicated threads only) |
| DI | a property on `OptunaStudyOptions`, the existing `AddOptunaStudyFactory(o => …)` pattern; no `IOptions<T>` |
| What the pool does above its capacity | nothing: the plain pool, documented only. No check, no warning, no `SetMinThreads`. The owner: "If user has achieved threadpool starvation, they should consider either dedicatedthreads which reuses created threads or strengthen the threadpool." |
| `nJobs` | Python's meaning on both executors: default 1, `-1` = `Environment.ProcessorCount` |
| Names | deferred, then settled on 2026-11-07 (§1.8) |

Considered and not chosen: a per-call argument only; a study setting only; accepting any `TaskScheduler`;
an open executor base class; raising the pool's minimum for the run; capping `nJobs` at the pool's size;
a warning above the pool's minimum; `-1` following the pool's minimum; parallel by default;
`IOptions<TrialExecutorOptions>` binding; `ThreadName`, `Priority` and `SharedThreads` settings; a
static-instances-only type; an immutable record; freeze-on-first-use.

### 1.3 API sketch (the names as settled in §1.8)

```csharp
namespace OptunaSharp;   // working; the namespace is deferred too

public enum TrialExecutorKind
{
    DedicatedThreads,    // default: OptunaSharp's own threads, reused across trials and Optimize calls
    ThreadPool,          // the .NET thread pool
}

public class TrialExecutorOptions        // not sealed: the owner wants it inheritable
{
    public virtual TrialExecutorKind Executor { get; set; } = TrialExecutorKind.DedicatedThreads;

    // Dedicated threads only: how long an idle worker waits for the next trial before its thread ends.
    // Today a fixed 30 s. Timeout.InfiniteTimeSpan keeps idle workers forever.
    public virtual TimeSpan ThreadIdleTimeout { get; set; } = TimeSpan.FromSeconds(30);

    // The process-wide default, read at the start of every Optimize call. Assigning null throws.
    public static TrialExecutorOptions Default { get; set; } = new();
}

public class Study
{
    // null until set: TrialExecutorOptions.Default applies. Not stored: LoadStudy starts with null.
    public TrialExecutorOptions? ExecutorOptions { get; set; }
}

// Both Optimize overloads gain one last optional parameter, after showProgressBar:
//     TrialExecutorOptions? executorOptions = null

public sealed class OptunaStudyOptions
{
    // When set, the DI study factory copies it into every study it builds (study.ExecutorOptions).
    public TrialExecutorOptions? DefaultExecutorOptions { get; set; }
}
```

### 1.4 Behavior

- **Resolution**, at the start of each `Optimize` call:
  `executorOptions argument ?? study.ExecutorOptions ?? TrialExecutorOptions.Default`.
- **Snapshot.** The resolved object's `Executor` and `ThreadIdleTimeout` are read once (one call of each
  getter, so a derived class's override is honored once) into an internal immutable value. Changing any
  options object afterwards affects only later calls.
- **Validation**, at the snapshot, whatever `nJobs` is: an undefined `Executor` value or a
  `ThreadIdleTimeout` that is zero or negative (other than `Timeout.InfiniteTimeSpan`) throws
  `ArgumentOutOfRangeException`. These are beyond-Optuna parameters, so .NET exceptions, not Python ones.
- **`nJobs: 1`** runs on the caller's thread, as today; the executor kind does not matter there.
- **Dedicated threads.** One `TrialWorkerScheduler` per distinct `ThreadIdleTimeout`, kept in a
  process-wide registry; today's `Shared` instance is the 30 s entry. Options with equal timeouts share
  one thread set, so reuse across studies and calls keeps working.
- **Thread pool.** `Task.Factory.StartNew(worker, CancellationToken.None,
  TaskCreationOptions.DenyChildAttach | TaskCreationOptions.PreferFairness, TaskScheduler.Default)`.
  `PreferFairness` queues each worker on the pool's global queue, so an `Optimize` called from a pool
  thread cannot inline its own trials through `Task.WaitAll`, and queued trials take turns with the
  host's own work. `ThreadIdleTimeout` is ignored: the pool manages its threads.
- **Unchanged on both:** Python's submit/wait loop, its error and stop rules, the GIL-aware waits
  (`MonitorGate.Wait`, P3-1), the worker hygiene (`InOptimizeLoop` restored, `CachedAllTrials` dropped,
  which matters more on pool threads the host reuses), and the `ExecutionContext` flow.
- **The pool's limits are documented, not handled** (owner's decision). The doc comment of
  `TrialExecutorKind.ThreadPool` and THREADING.md say, in substance: a trial blocks its thread for the
  whole trial. Once `nJobs` reaches the pool's minimum worker count (`ThreadPool.GetMinThreads`, the
  processor count by default), the pool adds threads only about once a second. StackExchange.Redis and
  gRPC calls, which complete on the pool, then stall or time out, and the host's other work waits behind
  queued trials. Measured on 32 cores: Redis timed out from about 40 workers, gRPC took 54.5 s at 96, and
  at most 34 trials ran at once (THREADING.md §3.6, §7.2). **If that happens, use `DedicatedThreads`,
  which reuses OptunaSharp's own threads, or raise the pool's minimum with `ThreadPool.SetMinThreads`.**

### 1.5 Implementation steps

1. Core: `TrialExecutorKind` and `TrialExecutorOptions`, fully XML-documented for the doc gate (the
   snapshot, inheritance, the pool's limits and the advice above).
2. Core: `Study.ExecutorOptions`.
3. Core: the `executorOptions` parameter on both `Study.Optimize` overloads and on
   `OptimizeLoop.Optimize`; resolution and snapshot; `StartWorker` dispatching on the snapshot.
4. Core: a `TrialWorkerScheduler` registry keyed by idle timeout (`Shared` = the 30 s entry); keep its
   counters for tests.
5. Core DI: `OptunaStudyOptions.DefaultExecutorOptions`; `OptunaStudyFactory` copies it into each study it builds.
6. Callers inside OptunaSharp: `OptunaSearchCV` creates and optimizes its own study with `nJobs`. It
   follows the static default and the study setting with no change; check that no other `Optimize(nJobs)`
   caller exists (the LightGBM tuner runs sequentially).
7. Docs (§1.7). Then the commit, the catalog's Changelog, and THREADING.md's status line.

### 1.6 Tests

- Resolution: the argument beats the study setting, which beats the static default; a static default
  changed between two calls reaches a study that never set its own; the DI factory copies
  `OptunaStudyOptions.DefaultExecutorOptions`; `CreateStudy` and `LoadStudy` leave `ExecutorOptions` null.
- Snapshot: changing the options object during a run does not change the running call; a derived class's
  overridden getters are read once.
- Pool: trials run on pool threads (`Thread.CurrentThread.IsThreadPoolThread`); the default stays off the
  pool (`NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool` keeps passing); an `Optimize(nJobs: 4)`
  called from a pool thread never runs a trial on that thread.
- Idle timeout: distinct timeouts get distinct thread sets, equal ones share; a short timeout retires
  threads and they restart on demand; zero and negative values are refused, infinite is accepted.
- `nJobs: -1` is the processor count on both executors; `nJobs: 1` runs on the caller whatever the options.
- Python semantics on the pool: the error-rule tests of `OptimizeConcurrencyTests` and
  `OptimizeEdgeCaseTests` run once per executor (`DataRow`).
- GIL: one `GilAwareMonitorTests` scenario on the pool (interop suite).
- The CI unit filter on both TFMs, the doc gate, and the examples compile gate.

### 1.7 Docs to update with Phase 1

- THREADING.md: the status line; §2 (pool workers in the threads table); §3.5 (both executors and the
  four levels); §3.6 (the decision, replacing "the owner's open decision"); §4 (thread-local hygiene on
  pool threads); §7.2 (the starvation advice); §10 (choosing an executor); §11 (the new tests); §12;
  §13 (the Python comparison).
- README: the header sentence that workers are "never taken from the .NET ThreadPool" becomes "by default".
- Site: `key-features/parallelization.md`, the FAQ's parallelization answer, and `storage.md`'s
  "StackExchange.Redis and the thread pool" paragraph.
- The catalog project doc.

### 1.8 Names

The owner deferred the final names on 2026-11-07, then settled them the same day, before the commit, after
Phase 1 was built with working names. Choices offered: keep the working names, rename only the DI
property, or move the two types to an `OptunaSharp.Threading` namespace. The owner chose to rename only the
DI property: `OptunaStudyOptions.TrialExecutor` became `OptunaStudyOptions.DefaultExecutorOptions`, which
matches its siblings (`DefaultDirection`, `DefaultStudyName`) and the other levels' *ExecutorOptions*.

The public names: `TrialExecutorKind` (`DedicatedThreads`, `ThreadPool`), `TrialExecutorOptions`
(`Executor`, `ThreadIdleTimeout`, `Default`), `Study.ExecutorOptions`, `Optimize(…, executorOptions:)`,
`OptunaStudyOptions.DefaultExecutorOptions`, all in namespace `OptunaSharp` (the DI type in
`OptunaSharp.Extensions`, as before). `TrialExecutorOptions` and `ThreadIdleTimeout` are the owner's own
names. `DedicatedThreads` is also the name of a StackExchange.Redis feature flag, so the docs always say
"StackExchange.Redis's `DedicatedThreads` flag" for that one.

### 1.9 Outcome (2026-11-07)

Built as designed in §1.2–1.4, with the names of §1.8:

- `src/OptunaSharp.Core/TrialExecutorOptions.cs`: `TrialExecutorKind` and `TrialExecutorOptions`. The two
  properties are virtual and use volatile reads and writes. The idle timeout is stored as 64-bit ticks, so a
  32-bit process never reads a torn value. Validation: an undefined kind; a timeout that is zero, negative
  (other than `Timeout.InfiniteTimeSpan`) or above `int.MaxValue` ms, `Monitor.Wait`'s limit.
- `src/OptunaSharp.Core/Internal/TrialExecutorSnapshot.cs`: the resolution and the once-per-call read. The
  pool branch is `TaskScheduler.Default` with `DenyChildAttach | PreferFairness`. The dedicated branch is
  `TrialWorkerScheduler.For(timeout)`, a `ConcurrentDictionary` registry; `Shared` is the 30 s entry.
- `Study.ExecutorOptions` (volatile), the `executorOptions` parameter on both `Study.Optimize` overloads and
  `OptimizeLoop.Optimize`, `OptunaStudyOptions.DefaultExecutorOptions`, and `OptunaStudyFactory` assigning it to
  every study it builds.
- Step 6 checked: `OptunaSearchCV` is the only other `Optimize(nJobs)` caller in the repo, and it follows
  the static default with no change.
- Tests: `TrialExecutorOptionsTests` (32 cases, every bullet of §1.6, plus the `ExecutionContext` flow on
  both executors, which no test pinned before) and the `GilAwareMonitorTests` pool scenario. The CI unit
  filter passed on both TFMs (1,560 per TFM before the context-flow test was added).
- Measured on the built code (32 CPUs, pool minimum 32; 192 trials of 1 s at `nJobs: 96`, ideal 2 s):
  dedicated threads ran 96 at once in 2.0–2.3 s, and the host's pool work waited at most 1 ms; the pool ran
  at most 33 at once in 6.1 s, and the host waited up to 2.0 s. THREADING.md §3.6 has the table.
- Docs: THREADING.md (status line, §2, §3.1, §3.5, §3.6, §4, §5.1, §7.2, §8, §10–§13), the README header,
  the site's parallelization page ("Which threads run the trials"), the FAQ's parallelization answer and
  `storage.md`'s concurrency bullets.

---

## Phase 2 — six follow-ups, after Phase 1 is finished

The owner chose all six on 2026-11-07. Each section gives why, the design, the sub-decisions still open
(asked when the item starts), the tests, and when it is done.

### 2.1 Close Open item 24.h: seeds, the terminator lock, the divergence test

- **Why.** THREADING.md §6.3: every OS seed has 32 bits. One million reseeds repeated 116 first draws, the
  birthday bound's 116.4; one million numpy reseeds repeated none. §6.6: `TerminatorCallback.Invoke` takes
  no lock while its evaluators (`RegretBoundEvaluator`, `EMMREvaluator`) share one `LazyRandomState`
  across concurrent `nJobs` callbacks. §3.3: a trial failed while its objective still runs (a heartbeat
  stopped) makes C# raise `ValueError: Cannot tell a FAIL trial.` where Optuna 5.0 raises
  `AssertionError('Should not reach.')`, and no test pins it.
- **Design.**
  - One internal helper fills four 32-bit words from `RandomNumberGenerator` and calls NumSharp's
    `RandomState.seed(uint[])` (128 bits). `LazyRandomState.SetRng` and every `ReseedRng` use it (13 files
    call `RandomNumberGenerator.Fill` today). Explicit seeds never reach this path, so seeded runs stay
    bit-identical.
  - `TerminatorCallback.Invoke` runs under a private lock, and THREADING.md's user workaround goes away.
  - A test pins the stale-trial divergence.
- **Open sub-decision.** Pin the divergence as `[Misaligned]`, or raise Python's `AssertionError`
  equivalent.
- **Tests.** The helper seeds from 128 bits (state keys differ across reseeds); seeded trajectories
  unchanged (the digest technique of the pass-3 review); concurrent terminator callbacks at `nJobs: 8`;
  the divergence test.
- **Done when** the three items are gone from THREADING.md §6.3, §6.6, §11 and §12 and from catalog Open
  item 24.h.

### 2.2 Redis journal: 19 → 11 round trips per trial

- **Why.** THREADING.md §7.3: the journal holds one monitor across every round trip of an operation, as
  Python's does. A trial with two parameters costs 19 Redis round trips (11 `GET`, 4 `SETNX`,
  4 `EVALSHA`), all serialized, so a Redis journal runs about 40–50 trials/s at any `nJobs`.
- **Design**, for standalone servers:
  - each sync becomes one `EVALSHA` of a read-only Lua script that returns `log_number` and the records
    after the last one read (today: one `GET` for the counter plus one `GET` per record);
  - the counter is seeded with `SETNX` once per backend instance, not before every append.
  - Keys, record bytes and the append script do not change, so Python workers on the same journal are
    unaffected.
  - With `useCluster: true` the keys sit in different hash slots, so no script can read them together;
    cluster mode keeps its per-key reads.
  - Applies to all three clients: StackExchange.Redis, NRedisStack, ServiceStack. Fewer round trips also
    stretch ServiceStack's free-tier request budget.
- **Measured** on the 2026-10-29 scratch build (1.1 ms round trip, a busy machine): 40 → 71 trials/s at
  `nJobs: 1`, 42 → 54 at `nJobs: 8`.
- **Open sub-decisions.** Pipeline the cluster-mode reads, or leave them; cap the records one script call
  returns (for a worker catching up on a long journal).
- **Tests.** Client round trips counted per trial (count client calls, not `INFO commandstats`, which also
  counts the commands inside a script); the journal contract tests on all three clients; the
  `JournalMixedFleetInteropTests` Redis case (Python and C# workers on one journal); ServiceStack runs kept
  within its free tier (the owner's rule: at most 6,000 requests per hour per process).
- **Done when** a trial costs 11 round trips on a standalone server, and THREADING.md §7.3 and the
  storage page say so.

### 2.3 Redis: `DedicatedThreads` on the connections OptunaSharp opens

- **Why.** THREADING.md §7.2: StackExchange.Redis 3.x completes synchronous calls on the thread pool. A
  host calling Ask/Tell from 64 concurrent `Task.Run` callers finished 624 of 640 trials in 92 s with
  timeouts; with the flag, 640 of 640 in 11.3 s. With 96 callers: 933 of 960 in 163 s, against 960 of 960
  in 18.2 s.
- **Design.** The StackExchange.Redis and NRedisStack backends' URL and `ConfigurationOptions`
  constructors call `ConnectionMultiplexer.SetFeatureFlag("DedicatedThreads", true)` before they connect.
  An `IConnectionMultiplexer` the caller passes in is left exactly as built. ServiceStack uses its own
  synchronous sockets and needs nothing.
- **Documentation** (the owner: "Make sure this is documented"), in the backends' XML remarks, the Redis
  integration page, `storage.md` and THREADING.md §7.2:
  - the flag is process-wide, so every StackExchange.Redis connection the process opens afterwards gets
    dedicated threads too;
  - it costs a reader and a writer thread per connection;
  - StackExchange.Redis reserves `SetFeatureFlag` for support guidance and hides it from IntelliSense;
  - to opt out, pass your own multiplexer.
- **To verify first.** Whether StackExchange.Redis 2.13.17, the `net8.0` build, has the flag and the same
  pool-bound reader; the 2026-10-29 analysis covered 3.1.31 only.
- **Open sub-decision.** On by default, or an opt-in constructor argument.
- **Tests.** The host scenario scaled down in the Storage suite (many pool-thread callers doing Ask/Tell):
  every trial completes, no `RedisTimeoutException`; an injected multiplexer is not affected.
- **Done when** the flag is set where decided, documented in all four places, and the host scenario is
  pinned.

### 2.4 Cancellation: a `CancellationToken` and Ctrl+C

- **Why.** THREADING.md §3.4: `Stop()` works only from inside an objective or a callback. Ctrl+C ends the
  process, and the trials in flight stay `RUNNING` in a persistent storage; a journal never fails them.
- **Design.**
  - `CancellationToken cancellationToken = default` on both `Optimize` overloads, after `executorOptions`.
  - Cancelling stops the scheduling like `Stop()`: the trials in flight finish and are stored, then
    `OperationCanceledException` (carrying the token) is thrown. This is Optuna's `n_jobs > 1` behavior on
    `KeyboardInterrupt`.
  - OptunaSharp installs no Ctrl+C handler itself, because that is process-wide. The docs show the
    `Console.CancelKeyPress` recipe: `e.Cancel = true; cts.Cancel();`.
- **Open sub-decisions.**
  - At `nJobs: 1`: let the running trial finish, or fail it as Python does on `KeyboardInterrupt` (which
    needs the objective to observe the token).
  - Expose the token on `Trial`, so objectives can abort a long evaluation.
  - A small Ctrl+C helper, or the recipe only.
  - The parameter name, with the deferred names.
- **Tests.** Cancel mid-run at `nJobs` 1 and 8: no trial left `RUNNING`, the exception type and token; an
  already-cancelled token runs no trial; `Stop()` still works; both executors.
- **Done when** THREADING.md §3.4 and §12 describe the token and the recipe.

### 2.5 `PymooSampler` on any thread (review item 19b)

- **Why.** Driven from a thread other than the one that built it, `PymooSampler` crashed in pymoo's
  `Algorithm.ask` with an access violation (`0xC0000005`) in 2 of 4 runs. Today it cannot run with `nJobs`
  above 1, nor from another thread.
- **First step: find the root cause.** The sampler rebuilds its pymoo algorithm inside every
  `SampleRelative` call (reconstruct and replay), so no pymoo object outlives a call. Something that is
  bound to a thread does: a candidate is state cached at import or construction (the constructor runs
  `Python.pymoo.check`).
- **Likely design:** one owner thread per `PymooSampler`. Every pymoo call is posted to it and awaited
  through `MonitorGate.Wait`, so a caller holding the GIL releases it. Calls are already serial under the
  sampler monitor, so the hand-off costs one thread switch per call and loses no parallelism.
- **Also relevant (found 2026-11-07, `d5c25f3`):** on macOS arm64, `AgeMoea` and `AgeMoea2` JIT-compile
  numba kernels on their first `tell`, which overflows the default stack of any .NET-created thread
  (`nJobs` workers on both executors, pool threads, test runners). CI now sets 16 MB default stacks. An
  owner thread created with a `maxStackSize` of 16 MB would remove that limit for `PymooSampler` too.
- **Open sub-decisions.**
  - An owner thread, or refusing calls from another thread with a clear error.
  - How the owner thread ends: `IDisposable`, or a background thread that retires when idle.
- **Tests.** `Optimize` from a thread other than the constructing one, 20 runs; `nJobs: 4` with
  `PymooSampler`, every trial complete.
- **Done when** item 19b is closed in the catalog and THREADING.md §8 and §12 no longer list it.

### 2.6 `MLNetEstimator` under `OptunaSearchCV(nJobs > 1)`

- **Why.** THREADING.md §12 lists it as untested: the sklearn-style search runs trials in parallel, and the
  ML.NET adapter has never run that way.
- **Design.** The test first: `OptunaSearchCV` with an `MLNetEstimator` at `nJobs: 4`. Every trial
  completes, and each trial's score equals a sequential run with the same parameters and seeds. Fix the
  adapter only if the test fails; shared ML.NET state is the usual suspect, for example one
  `PredictionEngine`, which ML.NET documents as not thread-safe.
- **Open sub-decisions.** None until the test has run.
- **Done when** the test is green on both TFMs and THREADING.md §10 and §12 are updated.

---

## Order

1. Settle the deferred names (§1.8). Done 2026-11-07.
2. Phase 1, in one commit series: §1.5 steps 1–7. Done 2026-11-07 (§1.9).
3. Phase 2, one item at a time, each asking its open sub-decisions first. Suggested order: 2.1 (small,
   low risk), 2.6 (a test that may need no fix), 2.3, 2.2 (the cross-language journal contract), 2.4 (new
   public API), 2.5 (needs a root cause first).
