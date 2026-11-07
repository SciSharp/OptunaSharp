# Plan 10 — Threading: a configurable `nJobs` executor, then seven follow-ups

> **Status: Phase 1 BUILT (2026-11-07, §1.9); Phase 2 DONE (2.1 to 2.7, 2026-11-07).** Phase 1, the executor, came first.
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

## Phase 2 — seven follow-ups, after Phase 1 is finished

The owner chose the first six on 2026-11-07, and added 2.7 the same day from a finding of 2.2. Each section gives why, the design, the sub-decisions still open
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
- **DONE 2026-11-07.** Outcome:
  - Sub-decision (owner, 2026-11-07): pin the divergence as `[Misaligned]`. Re-observed on Optuna 5.0.0 with an
    in-memory heartbeat storage: `AssertionError('Should not reach.')` whose context is the `ValueError`, and the
    `ValueError` itself under `python -O`. Test: `OptimizeEdgeCaseTests.TrialFailedAsStaleWhileItsObjectiveRuns_RaisesCannotTellAFailTrial`.
  - Seeds: one design refinement over the sketch above. NumSharp ships numpy's `SeedSequence`, so
    `Internal/OsEntropySeed.cs` ports numpy's own construction instead of calling `seed(uint[])`: 128 bits
    through `SeedSequence().generate_state(624)`, key[0] forced to `0x80000000`, the constructor's position 623,
    and a reseed that keeps the position and drops the cached Gaussian (`MT19937.__init__`,
    `_legacy_seeding(None)`, `RandomState.seed`). `LazyRandomState.SetRng` and the 12 Core `ReseedRng`
    overrides call it. `BoTorchSampler.ReseedRng` is unchanged: it draws a 31-bit torch seed, as Python's
    `RandomState().randint(int32 max)` does.
  - Terminator: `TerminatorCallback.Invoke` runs `ShouldTerminate` under a private lock entered through
    `MonitorGate` (GIL-aware); `Study.Stop` stays outside it.
  - Tests: `OsEntropySeedTests` (8, against numpy 2.4.2 output for injected entropy, plus 300,000 reseeds with
    no repeated draw) and `TerminatorTests.TerminatorCallback_UnderNJobs_RunsItsTerminatorOneCallAtATime`.
    Negative controls on the old code: 9–14 repeated draws per 300,000 32-bit reseeds (bound 10.5), and 8
    terminator calls at once without the lock.

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
- **DONE 2026-11-07.** Outcome:
  - Sub-decisions (owner, 2026-11-07, two rounds):
    - The counter's `SETNX` moves inside the append script ("Inside the append script"), rather than once per
      backend instance. The append stays atomic, and numbering still starts at 0 if the journal's keys are deleted
      under a live backend.
    - Cluster-mode reads stay key by key ("Leave as is").
    - The read cap became a question about snapshots. The owner's words: "Can't we maintain a latest or snpashots
      and so user gets from point X+diff to head ... We must maintain backwards compatability to optuna". Each
      language now keeps its own snapshot ("Own C# snapshot key"): OptunaSharp's JSON under
      `{prefix}:snapshot:optunasharp`, and Python's pickle stays alone in `{prefix}:snapshot`. A cold start of
      either language restores its own latest snapshot and reads only the records after it. One read script call
      returns at most 1,000 records (`RedisJournalProtocol.ReadLogsBatchLimit`), so a script never holds the
      server for long.
    - The snapshot interval is configurable ("make it configurable") and was measured ("Measure every how many
      trails it gets recommended to create snapshot"). The owner chose a default by read mode ("By read mode:
      100/1000"):
      - `JournalStorage(backend, int? snapshotInterval = null)`; null asks the backend through the new
        `IJournalSnapshotInterval`, then falls back to Optuna's 100;
      - the Redis backends answer 1000 on a standalone server and 100 in cluster mode;
      - `JournalStorage.SnapshotInterval` reports the value in use.
  - Round trips: 12 client commands per two-parameter trial (8 read-script syncs, 4 appends), against 20 before.
    The plan's "19 → 11" counted the server's statistics during `Optimize`. The profiler, counting the client's
    calls for an asked-and-told trial, gives 20 → 12; the saving is the same 8.
  - Throughput: 92–127 trials/s at `nJobs` 1 and 8, against 62–69, on a local Redis (0.66 ms median `PING`).
    A cold start over 400 trials took 24–49 ms instead of 312–391.
  - Compatibility: keys, record bytes and numbering are Python's. Python and C# workers still produce identical
    studies on one journal (`JournalMixedFleetInteropTests`), each client still round-trips with live Optuna,
    and a new cross-language test runs both languages' snapshots on one journal. Each language restores its own
    without the failed-restore warning both logged while they shared one key. An OptunaSharp snapshot an older
    version saved in Python's key is still found while OptunaSharp's own key is empty. Python's key is checked by its first byte (`GETRANGE 0 0`) before it is read, so a pickle there is never downloaded or handed to the OptunaSharp storage.
  - Snapshot measurements, at 20,000 trials: an interval of 100 made the writer 48% slower than no snapshots, and
    1000 made it 6% slower. A cold start took 1.19 s, 1.22 s and 1.30 s on a standalone server, and 1.17 s, 4.03 s
    and 53.8 s in cluster mode, which reads every record with its own `GET`. The snapshot's JSON is the
    bottleneck, which led to item 2.7.
  - Correction (found during 2.7): those probes ran with the JIT optimizer off, since `dotnet run file.cs`
    ignores a file's `#:property Configuration=Release`. Rerun in Release on the old code: 13.62, 11.10 and
    11.65 ms per trial at 100, 1000 and no snapshot, and standalone cold starts of 962, 944 and 1143 ms. After
    item 2.7's binary snapshot, the owner moved the standalone default to Optuna's 100 (see 2.7).
  - Tests: `RedisJournalCostAndThreadsTests` (round trips, the interval under the own key, Python's pickle read by its
    first byte only), the contract tests'
    batch, seeding, snapshot-key and interval cases on all three clients, `JournalStorageTests`'s interval cases,
    and `JournalRedisInteropTests.PythonAndCSharp_OnOneRedisJournal_EachKeepsItsOwnSnapshot`. ServiceStack's
    requests stay within its free tier: batching cut them too.

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
- **DONE 2026-11-07.** Outcome:
  - Sub-decision (owner, 2026-11-07): an opt-in constructor argument, defaulting to true ("Opt-in argument",
    "defaulting true"). It is `dedicatedThreads: true` on the URL and `ConfigurationOptions` constructors of the
    StackExchange.Redis and NRedisStack backends and on their DI `Add…JournalStorage(url, …)` methods. `false`
    leaves the flag as it is, and an injected multiplexer is never touched.
  - Verified first: StackExchange.Redis 2.x (2.13.17, and 2.12.14 under NRedisStack 1.5.0) has no such flag and
    silently ignores the name. It did not time out under the same load, but took 37.6 s for 64 callers, against
    8.2 s with the pool's minimum at 200 (THREADING.md §7.2 has the whole matrix).
  - Measured on `net10.0`, 64 callers × 10 trials on 32 cores: 634 of 640 trials in 37.0 s without the flag (6
    failed calls, 4 of them `RedisTimeoutException`), and 640 of 640 in 7.0 s with it.
  - Documented in the backends' XML remarks, the Redis integration page (a section of its own), `storage.md`, and
    THREADING.md §7.2 and §10.
  - Tests: `RedisJournalCostAndThreadsTests`. The owning constructors turn the flag on, `false` leaves it alone,
    and an injected multiplexer stays untouched; these report Inconclusive on 2.x. More thread-pool callers than
    twice the pool's minimum ask and tell without losing a trial, on 3.x and on 2.x.

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
- **DONE 2026-11-07.** Outcome:
  - Sub-decisions (owner, 2026-11-07, three HTML rounds):
    - "Token on Trial": `Trial.CancellationToken`. An objective that throws for it is stored FAIL; one that
      ignores it finishes.
    - The parameter is named `ct` (the owner's own answer, over the proposed `cancellationToken`):
      `Optimize(…, executorOptions, CancellationToken ct = default)` on both overloads.
    - Ctrl+C: the owner first asked "Which solution uses zero cpu until Ctrl+C is pressed". Measured: all of them;
      over 10 idle seconds the CPU matched no handler on Windows and Linux. The owner then chose a helper that
      also covers SIGTERM ("Helper incl. SIGTERM", commenting "Which is used with CLI exe launch": it serves
      console executables, and a Generic Host passes `ApplicationStopping` instead).
    - Its name: `StopSignals` (`using var signals = StopSignals.Listen(); … ct: signals.Token`).
  - Behavior:
    - The submit loop and every worker check the token before each trial, ahead of the stop flag.
    - Once it is cancelled no trial starts. The trials in flight finish and are stored, then
      `OperationCanceledException` is thrown carrying `ct`. An already cancelled token runs no trial.
    - An `OperationCanceledException` that escapes an objective while `ct` is cancelled stores that trial FAIL,
      whatever `catch` holds, and the callbacks skip it (Python's `KeyboardInterrupt`). A linked token is reported
      with the caller's token, the objective's exception being the cause. Another token's cancellation stays an
      ordinary failure.
  - `StopSignals` (Core, opt-in), over `PosixSignalRegistration`:
    - SIGINT and SIGTERM. The first signal is held, cancels the token and logs a warning; a second one takes its
      default action.
    - A throwing token callback cannot escape the handler.
    - `Dispose` unhooks; the token stays readable.
  - Python, re-observed on Optuna 5.0.0: at `n_jobs=1` an interrupt in the objective stores that trial FAIL, then
    `KeyboardInterrupt`. At `n_jobs=3` the trials in flight finish COMPLETE first.
  - Real-signal checks:
    - Linux container, `docker kill` with SIGTERM or SIGINT: graceful within about 1 s, every trial COMPLETE in a
      new storage. Two quick SIGINTs exit 130.
    - Windows, `CTRL_C_EVENT` raised on the demo's own console: graceful in 0.4 s; two presses exit `0xC000013A`.
      A process that inherits "ignore Ctrl+C" from its parent never sees Ctrl+C; documented.
  - Tests:
    - `OptimizeCancellationTests` (16 cases) and `StopSignalsTests` (8 cases, the real-SIGTERM case on Linux and
      macOS only).
    - Mutations caught: dropping the loop's token check fails 4 cases, and letting `catch` swallow the
      cancellation fails 2.
  - Docs: THREADING.md §3.4, §11 and §12; the site's parallelization page ("Stopping a run"); the callback recipe.

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
- **DONE 2026-11-07.** Outcome:
  - **Root cause** (the plan's first step):
    - Not the stack: a 16 MB thread crashed 6 of 6.
    - Not the constructing thread: built and driven on one new thread, 6 of 6. A pool thread crashed 6 of 6, the
      main thread 0 of 8, and the review's case (a new thread) 7 of 8.
    - The faulthandler stack pointed at scipy's pybind11 `cdist`, which pymoo's duplicate elimination calls inside
      `Algorithm.ask`; pymoo imports `scipy.spatial` lazily.
    - The mechanism: on a .NET thread, pythonnet's `Py.GIL()` creates a CPython thread state at the outermost scope
      and frees it at the end. pybind11 keeps the thread state of the thread on which its internals loaded, and uses
      it after it was freed.
    - Confirmed three ways. A worker that kept one thread state passed 8 of 8. Importing scipy on the main thread
      first passed 8 of 8. A bare probe (scipy `cdist`, torch `cholesky`) failed 5 of 5 each under churn (scipy
      crashed, torch hung) and passed 5 of 5 each with the state kept.
    - So BoTorch (torch) and any pybind11 package behind OptunaHub were exposed the same way. Pure-Python pymoo on
      Python threads ran with stacks down to 256 KB.
  - **Sub-decisions** (owner, 2026-11-07, three HTML rounds):
    - The first round was rejected ("I dont like the solutions, rethink this"): keep a thread state per caller
      thread in `PythonRuntime`, a `PymooSampler` owner thread, or both.
    - The second round chose "One Python thread", after measuring the hand-off: 0.5–0.8 µs a call for today's
      churn, 0.36 µs for a kept state, 1.1–1.3 µs for a hop to a dedicated thread. Those probes ran unoptimized
      (see 2.7); in Release: 0.55–0.8, 0.4 and 1.17–1.47 µs, and 1.45–1.56 µs through the shipped
      `PythonRuntime.Run`.
    - Separately, "16 MB worker stacks" for OptunaSharp's `nJobs` threads.
    - Refusing calls from other threads, the plan's other option, was no longer viable: one non-main thread
      crashed too.
  - **Design**:
    - `PythonRuntime.Run` hands every body to `PythonThread`: a long-lived background thread (`OptunaSharp Python`,
      16 MB stack) that starts the engine itself (it then holds CPython's main thread state for the process). With a
      host-started engine it takes a thread state of its own and keeps it. It runs each body under the GIL, one at a
      time.
    - Nested calls run inline. A caller that holds the GIL releases it while it waits.
    - The caller's `ExecutionContext` flows in, and exceptions come back with their stack traces.
    - `OptunaHubRuntime.EnsureQuiet` goes through `Run` too.
    - `TrialWorkerScheduler` threads reserve `WorkerStackSize` = 16 MB.
    - The rule a body must follow (never wait for a thread that waits for the Python thread) is documented. The
      bridges' bodies take only their own locks and the storages'.
  - **Checks** (the 19b probe, 8 runs per mode): a new thread, a pool thread, built and driven on one new thread,
    `nJobs: 4` and the main thread all passed 8 of 8. AGE-MOEA and AGE-MOEA-2 passed 4 of 4 on a new thread and at
    `nJobs: 4`. scipy and torch through `PythonRuntime.Run` from a worker passed 3 of 3.
  - **Tests**:
    - `PythonThreadTests` (Interop suite, 9 cases): one Python thread whose `threading.local` survives calls from
      six .NET threads, nested `Run` inline, a caller holding the GIL, context flow, exceptions, the 16 MB stack
      (Windows), `PymooSampler` off its thread and at `nJobs: 4`, torch from several threads, and the P3-1 shape
      through `Run`.
    - `TrialExecutorOptionsTests.DedicatedThreads_Reserve16MBOfStack` (Windows).
    - Negative control on the old `PythonRuntime`: the thread-state test and the stack test fail (different Python
      threads, a lost `threading.local`, a 1.5 MB stack).
  - **Docs**: THREADING.md §2, §5.2, §8, §10, §11 and §12; the site's pythonnet page ("The Python thread"), the Python
    integrations index, the pymoo page and the pymoo README.

### 2.6 `MLNetEstimator` under `OptunaSearchCV(nJobs > 1)`

- **Why.** THREADING.md §12 lists it as untested: the sklearn-style search runs trials in parallel, and the
  ML.NET adapter has never run that way.
- **Design.** The test first: `OptunaSearchCV` with an `MLNetEstimator` at `nJobs: 4`. Every trial
  completes, and each trial's score equals a sequential run with the same parameters and seeds. Fix the
  adapter only if the test fails; shared ML.NET state is the usual suspect, for example one
  `PredictionEngine`, which ML.NET documents as not thread-safe.
- **Open sub-decisions.** None until the test has run.
- **Done when** the test is green on both TFMs and THREADING.md §10 and §12 are updated.
- **DONE 2026-11-07, no fix needed.** `MLNetEstimatorNJobsTests.OptunaSearchCV_NJobs4_ScoresEveryFoldAsASequentialEvaluation`
  (Integration.ML.NET suite): 16 trials at `nJobs: 4` over one `MLNetEstimator` (FastTree regression, three folds).
  Every trial completes, no fold score is NaN, and every fold score equals a sequential evaluation through a
  separate context and estimator. The test also checks that the trials overlapped in time (summed durations above
  1.5× the wall span), so the comparison runs on real concurrency. Green 3 runs × 2 TFMs. Found while writing it:
  the estimator turns any fit or score exception into the fold's error score (NaN), so a concurrency fault would
  never raise; THREADING.md §10 now says so, along with the factory and scorer's thread-safety requirement and the
  untested case of trainers that draw from the shared context's random source.

### 2.7 A faster journal snapshot codec (added 2026-11-07)

- **Why.** Item 2.2's measurements: OptunaSharp's snapshot of 20,000 trials is 10.6 MB of JSON.
  - Restoring it takes 0.75–0.95 s. `PyJson.Loads` alone takes 0.48–0.58 s, where `JsonDocument.Parse` takes
    0.05 s.
  - Saving it takes 0.49–0.61 s, under the storage's monitor, so the writing worker stalls.
  - Replaying 80,001 records with batched reads takes 0.84–0.97 s, mostly parsing their JSON.
  - So a snapshot buys a standalone cold start little (1.19 s against 1.30 s), and the default interval had to
    trade writer cost against cluster-mode cold starts.
- **Design** (to be refined when the item starts):
  - write and read OptunaSharp's snapshot with `Utf8JsonWriter`/`Utf8JsonReader`. The snapshot is C#-only, so its
    format may change, but v1 snapshots must stay readable;
  - speed up the parse of journal records, which stay Python's JSON (`json.dumps`, with bare `NaN` and
    `Infinity` tokens that `System.Text.Json` rejects), keeping `PyJson`'s grammar exactly;
  - goal: restore and save several times faster, so snapshots pay off on standalone servers too.
- **Tests: very high parity coverage, the owner's condition** ("But you must do very high parity test
  coverage"):
  - every record kind Python writes, parsed by the new reader into exactly what `PyJson` gives;
  - Python-written journals (`json.dumps` of every value kind: NaN, ±Infinity, -0.0, subnormals, huge and tiny
    floats, big integers, unicode escapes, surrogate pairs, nested attrs) replayed into identical studies;
  - property tests of random values round-tripping;
  - v1 snapshots written by the current code restored identically;
  - mixed fleets with Python;
  - the measurements repeated.
- **Open sub-decisions.** Asked when the item starts.
- **Done when** restore and save are measured several times faster, the parity suite is green on both TFMs,
  and THREADING.md §7.3 and §12 are updated.
- **DONE 2026-11-07.** Outcome:
  - Measuring first found that every probe of this plan had run with the JIT optimizer off (`dotnet run
    file.cs` ignores `#:property Configuration=Release`; `dotnet run -c Release file.cs` does not), so the
    "Why" above overstates the costs. In Release, at 20,000 trials: saving 437 ms, restoring 643 ms,
    replaying 100,001 records 537 ms.
  - Prototypes on the same snapshot (Release): reading it took 392 ms through PyJson, 122 ms through a
    byte-level reader with PyJson's grammar, 80 ms as binary; writing it 106, 60 and 38 ms.
  - Owner's decisions:
    - round 13: **"Binary v2 format"**, over keeping v1 JSON with new code, and **"Share one object"** for
      distributions during replay and restore;
    - round 14, after the Release re-measurement below: **"Optuna's 100 everywhere"**, so a standalone Redis
      server snapshots every 100 ids like cluster mode and like Python, instead of 1000.
  - Design:
    - `PyJsonUtf8` (Core): PyJson's grammar on UTF-8 bytes. For any input it returns exactly what
      `PyJson.Loads(Encoding.UTF8.GetString(bytes))` returns, or throws the same, because it takes PyJson's
      decision at every position and hands anything it cannot finish to PyJson. Containers are built at their
      final size from a reusable value stack, and repeated keys and small integers are reused. The file
      journal and all three Redis clients read records through it (ServiceStack through its string overload,
      which sends text with surrogates to PyJson directly).
    - The file journal reads 64 KB at a time instead of a byte at a time.
    - `JournalDistributionCache`: the same distribution JSON gives the same object, for `FloatDistribution`,
      `IntDistribution` and `CategoricalDistribution` exactly; the deprecated types are built per use (their
      FutureWarning fires as before), `DiscreteUniformDistribution` with its public `Q` setter is never
      shared, and at most 4,096 texts are kept.
    - The v2 snapshot: header `OSJS`, version 2, flags 0; each distinct distribution JSON once in a table the
      trials index; studies and trials in their dictionaries' order; doubles as eight bytes, integers as
      LEB128 (zigzag for signed ones), strings as Latin-1 or UTF-16 code units, datetimes as UTC ticks cut to
      microseconds as v1's text was, attributes as tagged values of PyJson's object model (a value a record
      cannot carry goes through JSON first, as v1 wrote it). A restore canonicalizes NaN as v1's `NaN` token
      did, builds every study and trial with v1's arguments, checks counts against the bytes left, and turns
      any malformed input into "no snapshot". v1 still restores (now through `PyJsonUtf8` and the cache), and
      v1 is also the fallback writer when the v2 writer fails, which keeps v1's outcome. Older OptunaSharp
      versions cannot read v2; they log the failed-restore warning and replay the whole journal.
    - `RedisJournalProtocol.StandaloneSnapshotInterval` is 100 (round 14); the per-mode constants and
      `IJournalSnapshotInterval` stay.
  - Measured (Release, 20,000 trials with float, int and a string attribute, one session, in process):

    | | v1 JSON before | v2 binary after |
    |---|---|---|
    | snapshot save | 437 ms | 19.5 ms |
    | snapshot restore | 643 ms | 38.7 ms |
    | cold start, snapshot 1,000 trials back + 4,000 records | 650 ms | 46 ms |
    | replay of 100,001 records, no snapshot | 537 ms | 367 ms |
    | cold start of a journal file | 715 ms | 390 ms |
    | snapshot size | 11.1 MB | 1.6 MB |

  - On Redis (Release, local server, the same study), which round 14 rested on:

    | Interval | Writer, ms per trial | Cold start, standalone | Cold start, cluster mode |
    |---|---|---|---|
    | 100 | 9.76 | 109 ms | 0.43 s |
    | 1000 | 14.81 (builds ran meanwhile) | 170 ms | 3.3 s |
    | none | 10.74 | 927 ms | 70.8 s |

  - Item 2.5's hop cost, re-measured in Release: 1.45–1.56 µs a call through `PythonRuntime.Run`.
  - Tests: `PyJsonUtf8Tests` (149 cases), `JournalSnapshotV2Tests` (21), `JournalCodecParityInteropTests` (4,
    live Optuna). Four deliberate bugs were caught: a restore keeping UTC instead of local time, a repeated key
    keeping its first value, parameters written in their external representation, studies written in id order.

---

## Order

1. Settle the deferred names (§1.8). Done 2026-11-07.
2. Phase 1, in one commit series: §1.5 steps 1–7. Done 2026-11-07 (§1.9).
3. Phase 2, one item at a time, each asking its open sub-decisions first. Suggested order: 2.1 (small,
   low risk; done 2026-11-07), 2.6 (a test that may need no fix; done 2026-11-07, none needed), 2.3 and 2.2
   (the cross-language journal contract; done together 2026-11-07), 2.4 (new public API; done 2026-11-07), 2.5
   (root cause first; done 2026-11-07), 2.7 (the owner's choice: after 2.4 and 2.5; done 2026-11-07).
