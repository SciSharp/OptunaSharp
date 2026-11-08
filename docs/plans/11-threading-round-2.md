# Plan 11 — Threading, round 2: sampler per worker, OpenBLAS threads, locked log handlers, and the async question

> **Status: DONE (2026-11-08).** The owner chose these on 2026-11-08 from four questions that followed the GIL fix of
> catalog Open item 28 (commit `889a32e`). Three were built; the fourth, a full async stack, was evaluated as asked
> and not built. [`docs/THREADING.md`](../THREADING.md) describes what the code does: §6.3 the samplers, §6.6
> logging, §9 OpenBLAS, §12 the limits.

---

## 1. The decisions (owner, 2026-11-08)

| Question | Answer, in the owner's words where given |
|---|---|
| Sampling under `nJobs` | "Sampler per worker", with: "Make it an enum configurable in TrialExecutorOptions, Dash-list our samplers and do they support concurrency (no duplicate sample producing) and if not, lock access to sampler. Default to same as optuna for parity" |
| OpenBLAS threads under `nJobs` | "Opt-in setting", with: "When null (nullable type) On nJobs == 1 use default settings which OpenBLAS uses multithreading but nJobs != 1, then use single-threaded OpenBLAS" |
| User code under `nJobs` (log handlers) | "I am thinking of lock log handlers. But i want you to check if the handlers are concurrently safe and so require no lock" |
| Async | "Full async stack", with: "Evaluate how much is that and present a shortform dash list" |

## 2. Sampler per worker — `TrialExecutorOptions.SamplerConcurrency`

- `enum SamplerConcurrency { Shared = 0, PerWorker = 1 }`, a fourth snapshot value of `TrialExecutorOptions`
  (read once per call, validated like the others). `Shared` is the default: Optuna's model.
- `PerWorker`: an `Optimize(nJobs > 1)` call keeps a `SamplerWorkerPool`. A worker rents an instance made from the
  study's sampler by the internal `BaseSampler.CreateWorkerInstance()`, reseeds it, binds it to its thread
  (`Study.ThreadLocal.SamplerOverride`), runs its trial, restores the thread and returns the instance. The `Trial`
  binds the instance at construction; the tell on the worker thread reaches it (`Study.SamplerForCurrentThread`).
- "Lock access to sampler" for the rest: a sampler whose `CreateWorkerInstance` returns null keeps the shared
  instance and its monitor, one call at a time. That is every sampler outside the list below, every sampler outside
  OptunaSharp (the member is internal) and every subclass of a listed sampler.
- The list was decided by measurement, 2026-11-08: eight per-worker instances on one study, three rounds; a sampler
  takes part only with no repeated point in any round. Taking part: `RandomSampler`, `TPESampler` (both modes),
  `CmaEsSampler`, `VkdCmaSampler`, `GridSampler`, `PartialFixedSampler` over one of them. The table of the others,
  with their counts, is in THREADING.md §6.3; GP repeated one point in eleven rounds and stays shared.
- What it buys, measured at `nJobs: 8`: multivariate TPE over 10 parameters 1.95 s → 1.20 s; 40-dimensional CMA-ES
  3.68 s → 1.46 s; nothing when the objective dominates (univariate TPE, 20 ms objective, 1.56 s → 1.54 s).
- Pinned by `SamplerConcurrencyModeTests`.

## 3. OpenBLAS threads — `TrialExecutorOptions.BlasThreads`

- `int?`: `null` is the automatic policy the owner described (OpenBLAS untouched at `nJobs: 1`, one thread while
  several workers run); a positive count is held for any call. Zero and negative counts are refused.
- `Internal/BlasThreadPolicy` holds the counts of the running calls; the smallest is in effect, the original is
  restored by the last release, and the native setting changes only when the smallest count does.
- Measured: a 40-dimensional CMA-ES at `nJobs: 8` took 5.22 s with OpenBLAS's 24 threads and 3.58 s with one
  (4.50 s at `nJobs: 1`); a GP study showed no difference.
- Pinned by `BlasThreadPolicyTests`.

## 4. Log handlers — checked, then locked

- The check (8 threads × 5,000 failed-trial warnings with exceptions): the default handler over `Console.Error` kept
  only 3,698 of 40,000 records whole; a `StreamLogHandler` over a plain `StreamWriter` (the CLI's `--log-file`) kept
  412 and mangled 48,247 lines. They are not safe, so they get a lock.
- Built as Python has it: a lock per handler instance around `Emit` (`Handler.handle`), one write and one flush per
  record (`StreamHandler.emit`, issue 35046). Afterwards 40,000 of 40,000 whole in both cases.
- Found on the way: `AddHandler` registered a handler twice, so it received every record twice; Python's
  `addHandler` ignores the second add. Fixed.
- Pinned by `LoggingTests` and `InfrastructureConcurrencyCoverageTests`.

## 5. A full async stack — evaluated, not built

What "async all the way" would take, counted on 2026-11-08:

- `BaseStorage`'s 29 abstract and virtual members, each with a `…Async` twin, in 6 storages (in-memory, journal,
  cached, SQLite, EF Core, gRPC) and 4 journal backends (file, three Redis clients).
- 32 Core samplers and 4 bridged ones, 9 pruners and 22 `Suggest*` methods, since a sampler reads the storage.
- 140 `lock` statements and 19 `MonitorGate` entries: a monitor is owned by a thread and cannot be held across an
  `await`, so every lock that guards I/O becomes an async lock, and the lock order of THREADING.md §5.2 is rebuilt.
- The GIL design (§8) is per OS thread (CPython thread states), while an `await` resumes on any thread: every Python
  bridge would need its own thread affinity again.
- 321 test files assume synchronous calls; the byte-parity suites assume sequential order.
- It moves the port away from Optuna's structure, which is synchronous everywhere (principle 2 of CLAUDE.md).
- Size: a rewrite of every layer's I/O path, months of work rather than days.
- Cheaper, if async objectives are the need: an `OptimizeAsync` overload that awaits a `Func<Trial, Task<double>>`
  on the existing worker threads (no `SynchronizationContext` there), which touches neither storages nor samplers.
  Not built; it is listed in THREADING.md §12.
