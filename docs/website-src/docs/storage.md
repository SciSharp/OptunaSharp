# Storage

A **storage** is where a study keeps its trials, parameters, attributes, and results. It is the
persistence layer behind every [`Study`](study-and-trials.md): samplers and pruners read history from
it, and `optimize` / `ask` / `tell` write to it. Choosing a storage decides whether a run is
in-process and ephemeral, survives a crash, or is shared across processes, machines, and even
languages.

OptunaSharp ports `optuna.storages`. All storage types live in the **`OptunaSharp.Storages`**
namespace (journal backends in **`OptunaSharp.Storages.Journal`**).

## Choosing a storage

| Storage | Persists? | Concurrency | Cross-language | Package |
|---------|-----------|-------------|----------------|---------|
| `InMemoryStorage` | No (process lifetime) | Single process | — | Core |
| `JournalStorage` + `JournalFileBackend` | Yes (a file) | Multi-process, multi-host (shared file) | **Yes** — Python Optuna reads/writes the same journal | Core |
| `JournalStorage` + a Redis client backend | Yes (Redis) | Distributed workers | Shared journal logs; runtime-specific snapshots | [Choose a Redis adapter](../integrations/storage/redis/index.md) |
| `RDBStorage` (SQLite) | Yes (a `.db` file) | Multi-process (one host) | **Yes** — Python Optuna opens the same SQLite file | `OptunaSharp.Storage.Sqlite` |
| `OptunaSharp.Storages.EFCore.RDBStorage` | Yes (PostgreSQL / MySQL) | See the [trial-creation concurrency limit](../integrations/storage/ef-core.md#concurrency-heartbeats-and-schema-lifecycle) | Optuna-compatible head schema | `OptunaSharp.Storage.EFCore` |
| `GrpcStorageProxy` + `GrpcProxyServer` | Via the fronted backend | Distributed workers (over the network) | **Yes** — interoperates with Python's gRPC proxy | `OptunaSharp.Storage.Grpc` |
| `CachedStorage` | Wraps another | Read-through cache over a backend | Inherited | Core |

The [Storage integrations](../integrations/storage/index.md) section documents SQLite, the
implemented EF Core PostgreSQL/MySQL backend, all three Redis client adapters, and gRPC. Construct
these backends directly: Core's `StorageFunctions.GetStorage` accepts a storage instance but does
not resolve an RDB URL merely because its satellite package is installed.

## In-memory (the default)

When you create a study without naming a storage, you get an `InMemoryStorage`. It is the fastest
option and needs no configuration, but everything is lost when the process exits.

```csharp
using OptunaSharp;

// No storage argument -> InMemoryStorage.
var study = Optuna.CreateStudy();
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);
```

A long study in memory can keep its finished trials compactly with `new InMemoryStorage(compactFinishedTrials:
true)`: a 100,000-trial CMA-ES study over 40 parameters retains 257 MB instead of 1,288 MB, with the same trials.
Finished trials then return read-only views instead of dictionaries; [Performance and
Compatibility](performance.md#memory-compact-finished-trials) lists what that changes.

## Journal file storage (persistent, shareable)

`JournalStorage` records every change as an appended JSON line; `JournalFileBackend` stores that
journal in a plain file. Because the format is line-delimited JSON over ordinary file I/O, the file is
**byte-compatible with Python Optuna's `JournalFileBackend`** — the same file can be opened, resumed,
and appended to from either runtime with no database or server in between.

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("study.log"));

// loadIfExists resumes the study if the journal already contains it.
var study = Optuna.CreateStudy(storage: storage, studyName: "sphere", loadIfExists: true);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 50);
```

Run this twice, or from several processes at once, and the trials accumulate in `study.log`.

### File locking

Concurrent writers coordinate through a lock file. Two implementations are provided, matching Optuna:

- `JournalFileSymlinkLock` — Optuna's default; uses a symlink, which on Windows needs Developer Mode
  or the symbolic-link privilege.
- `JournalFileOpenLock` — uses exclusive file creation; useful on Windows when symlink privileges
  are unavailable, provided the filesystem supports the required exclusive-create semantics.

When no lock is passed, the backend uses the symlink protocol and switches to the open-file protocol by
itself if this process cannot create a symlink next to the journal. Python Optuna raises in that case.
Both protocols take the same lock path, so mixed C# and Python workers still exclude each other.

```csharp
var backend = new JournalFileBackend("study.log", new JournalFileOpenLock("study.log"));
```

Both accept an optional `gracePeriod` (seconds, default `30`) after which a stale lock left by a
crashed process is reclaimed.

Both locks decide "another worker holds the lock" from the error code of the failed create, as Python
checks `errno == EEXIST`, and wait it out. Any other failure, such as a missing directory or no symlink
privilege, is raised. A waiter retries with a sleep that doubles from 1 ms up to 16 ms; Python's
cap is 1 s, which left the journal idle under contention. Exclusion is unaffected, so Python and C#
workers still share one journal.

## Redis journal storage (distributed)

For distributed workers sharing state through Redis, install a concrete client adapter such as
**`OptunaSharp.Storage.StackExchange.Redis`**. `OptunaSharp.Storage.Redis` is the shared protocol
package, not a directly constructible URL client. See the [Redis integrations](../integrations/storage/redis/index.md)
for StackExchange.Redis, NRedisStack, and ServiceStack.Redis configuration and client lifetimes.

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new StackExchangeJournalRedisBackend("redis://localhost:6379/0"));
var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
```

The adapters support `redis://` and `rediss://` (TLS); Unix-socket support and cluster configuration
depend on the client. A `prefix` isolates a journal namespace that may contain several studies.
Journal logs can be shared with Python Optuna. Each language keeps its replay snapshot under a key of
its own (`{prefix}:snapshot` for Python's pickle, `{prefix}:snapshot:optunasharp` for OptunaSharp's own binary format),
so a new worker of either language restores its own latest snapshot and reads only the records after it.
The [Redis guide](../integrations/storage/redis/index.md) explains those boundaries and how often each
backend snapshots.

## SQLite RDB storage (a persistent database file)

For a persistent, single-host database, install **`OptunaSharp.Storage.Sqlite`** and use `RDBStorage`
(namespace `OptunaSharp.Storages`). It writes Optuna's exact SQLite schema (alembic version and all), so
**Python Optuna can open the same `.db` file** — and OptunaSharp can open one Python wrote.

```csharp
using OptunaSharp;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///optuna.db");
var study = Optuna.CreateStudy(storage: storage, studyName: "sphere", loadIfExists: true);
```

`RDBStorage` implements `ICachedStorageBackend`, so `StorageFunctions.GetStorage` (and `Study`) front it
with a `CachedStorage` automatically, mirroring Optuna's `get_storage`.

## gRPC storage proxy (distributed, cross-language)

The **gRPC storage proxy** lets many workers — on different machines, in different languages — share one
storage over the network. A lightweight **client** forwards every storage operation to a **server** that
fronts a real backend (typically the SQLite `RDBStorage`). Install **`OptunaSharp.Storage.Grpc`**; the
client is `GrpcStorageProxy` and the server is `GrpcProxyServer`, both in `OptunaSharp.Storages.Grpc`. It
is a faithful port of `optuna.storages.GrpcStorageProxy` / `run_grpc_proxy_server`, experimental since
Optuna v4.2.0. This is a summary — see the dedicated [gRPC Storage Proxy](../integrations/storage/grpc/index.md) section for
the full client and server reference.

### The server

Run a server in front of any storage. Wrap an RDB backend with `StorageFunctions.GetStorage` (as Optuna
advises) so the server keeps an in-process cache:

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

var backend = new RDBStorage("sqlite:///optuna.db");
var storage = StorageFunctions.GetStorage(backend);
GrpcProxyServer.RunGrpcProxyServer(storage, host: "0.0.0.0", port: 13000);  // blocks
```

Need a non-blocking server (tests, or embedding it in a host)? `GrpcProxyServer.MakeServer(...)` returns
a startable `WebApplication`:

```csharp
var app = GrpcProxyServer.MakeServer(storage, "localhost", 13000);
await app.StartAsync();
// ... work ...
await app.StopAsync();
```

The server self-hosts on Grpc.AspNetCore (Kestrel) over cleartext HTTP/2, matching grpc's insecure port.

### The client

`GrpcStorageProxy` is a `BaseStorage` — pass it to `Optuna.CreateStudy` / `Optuna.LoadStudy` like any
other storage:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy(host: "server-host", port: 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));

var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);
```

Many clients can point at one server and optimize the same study concurrently — the distributed-worker
pattern the proxy exists for. Each client keeps an incremental trial cache (it re-fetches only new or
still-unfinished trials), so re-reading a growing study stays cheap.

### Configuring the connection

Beyond Optuna's host/port, the proxy exposes the full Grpc.Net.Client and Grpc.AspNetCore surface — the
on-wire protobuf is fixed by `api.proto`, so byte-compatibility holds however you connect. The **client**
has four constructors: `(host, port)`; `(host, port, Action<GrpcChannelOptions>)` to tweak the derived
options; `(address, GrpcChannelOptions)` for any address plus full options (e.g. an `https://` address
with TLS credentials); and `(GrpcChannel)` to inject a channel you built and own (the proxy never
disposes it). The **server**'s `MakeServer` / `RunGrpcProxyServer` take optional `configureGrpc`
(`GrpcServiceOptions`: interceptors, compression, message sizes) and `configureBuilder` (full
`WebApplicationBuilder` access: TLS, authentication, extra services). To add the proxy to an ASP.NET Core
app you already run, use the `services.AddOptunaStorageProxy(storage)` and `endpoints.MapOptunaStorageProxy()`
extensions instead of the self-contained host.

### Cross-language

The proxy speaks Optuna's own `api.proto`, so the two runtimes interoperate on the wire in **both
directions**: an OptunaSharp client can drive a Python `run_grpc_proxy_server`, and a Python
`GrpcStorageProxy` can drive an OptunaSharp server.

```python
# Python server; OptunaSharp (C#) clients connect to it
from optuna.storages import run_grpc_proxy_server, get_storage
run_grpc_proxy_server(get_storage("sqlite:///optuna.db"), host="0.0.0.0", port=13000)
```

Storage errors map through gRPC status codes exactly as Optuna does, so a missing study raises
`KeyError`, a duplicate name a `DuplicatedStudyError`, a write to a finished trial an
`UpdateFinishedTrialError`, and an invalid argument a `ValueError` — on either side of the wire.

> Like Optuna, the gRPC proxy does **not** support `JournalStorage`; put the SQLite `RDBStorage` behind
> the server. The transport is insecure (cleartext HTTP/2), matching Optuna — keep it behind your own
> network boundary.

## Concurrency: workers, studies and languages on one journal

A journal (a file or a Redis prefix) holds any number of studies. Workers can share it in every
combination below, and each study comes out complete: trial numbers are contiguous from 0, every
trial finishes, and Python replays the journal exactly as C# does.

| Setup | How it is synchronized |
|---|---|
| One `JournalStorage`, many threads (`Optimize(nJobs: …)`, your own threads, Ask/Tell) | The storage's monitor serializes its operations |
| Several `JournalStorage` instances in one process, on one journal | The journal lock (file) or Redis's atomic append, exactly as between processes |
| Several `JournalStorage` instances over one backend instance | As above; the file backend's offset cache is thread-safe |
| Several processes, C# and Python mixed | The same lock path and protocol as Python Optuna's `JournalFileBackend` (file), or Redis |
| Several studies at once, on shared or separate storages | Study and trial ids are assigned in journal order; numbering is per study |
| Concurrent same-name `CreateStudy(..., loadIfExists: true)` | One study is created; every caller loads it. Without `loadIfExists`, the others get `DuplicatedStudyError` |
| Concurrent workers popping `EnqueueTrial` trials | Each queued trial runs exactly once |

C# and Python workers can also share a sampler's state. `CmaEsSampler` keeps its optimizer in trial
system attributes as Optuna does, `pickle.dumps(optimizer).hex()` of a cmaes `CMA`, `SepCMA` or
`CMAwM`. Each language continues the state the other stored. Two workers taking turns on one study
produce exactly the trials two Python workers produce from the same seeds, and the stored states are
byte-identical to cmaes 0.13.1's. The exception is the history entries cmaes leaves uninitialized
(`np.empty`) until its first generations fill them. Two conditions apply:

- **`with_margin=True`.** The Python workers need cmaes 0.13.1, which changed `CMAwM`'s state; an
  older `CMAwM` state raises a `ValueError` naming the version. `CMA` and `SepCMA` states are the same
  in every cmaes Optuna supports (0.12.0 and later).
- **Studies started before 2026-10-29.** OptunaSharp stored its own JSON there, which a Python worker
  cannot read (`ValueError: non-hexadecimal number found in fromhex()`). C# still reads it, and stores a
  pickle at its next generation; Python workers can join from then on.

What each layer guarantees within one process:

- **Sampler and pruner.** Every call a study makes into its sampler (sample, infer, before/after trial,
  reseed) and pruner (prune, filter) runs under that instance's monitor. Concurrent trials never draw
  the same random numbers or corrupt a sampler's caches. The cost: model-based sampling for concurrent
  trials is serial, while the objectives still run in parallel. Single-threaded studies never contend,
  so seeded runs stay bit-identical to Optuna. An unseeded sampler seeds itself from OS entropy, as
  numpy does, so samplers created together never share a stream: not in one process, and not in
  worker processes or parallel `optunasharp ask` calls started at the same moment.
- **One sampler instance per study.** The monitor makes one sampler safe across threads, not across
  studies. A relative sampler caches its study's search space: TPE (multivariate by default in a
  single-objective study), CMA-ES, GP, NSGA-II/III and the other samplers built on
  `IntersectionSearchSpace`. Given a second study, it raises ``ValueError: `IntersectionSearchSpace` cannot
  handle multiple studies.``, exactly as Python's does. Give each concurrent study its own sampler.
  `AddOptunaSampler(factory)` does this by default: it registers the sampler as transient, so the study
  factory builds a new one per study.
- **`nJobs`.** Workers run on dedicated threads by default, as Python's `ThreadPoolExecutor` does; the
  .NET thread pool is an opt-in ([which threads run the trials](key-features/parallelization.md#which-threads-run-the-trials)).
  The rules below hold on both. When one trial
  raises an exception that is not caught, `Optimize` waits for the trials still running and then
  raises that original exception. Errors among the final trials are not raised, as in Python. They are
  still logged and stored as `FAIL`. `nJobs` of 0 or below -1 raises Python's
  `ValueError: max_workers must be greater than 0`.
- **Ending a run.** A timeout, `Stop()` or `MaxTrialsCallback` stops scheduling, and the trials already in
  flight finish, so an `nJobs` run can end up to `nJobs - 1` trials past the limit, as in Python.
  `Stop()` works only from an objective or a callback. Called from any other thread it raises Python's
  `RuntimeError` and the run goes on: to cancel from outside, set a flag that a callback checks. A failing
  sampler `BeforeTrial` is raised whatever `catch` holds and leaves its trial `RUNNING`, as in Python.
- **One trial, several threads.** A `Trial` serializes its own operations. When several threads report
  the same step, the first value wins, as Optuna documents (Python's own `Trial` races there). Suggesting
  one name from several threads samples it once.
- **Two tells of one trial.** Tells of one trial through one storage instance, from any thread and from
  the optimize loop, act in turn. Exactly one stores its result and reaches the sampler's `AfterTrial`.
  The others are refused with `ValueError: Cannot tell a COMPLETE trial.`, or return the stored trial
  with `skipIfFinished: true`. That is what Python shows in practice, and it is what a reaper that fails
  slow trials relies on. Before 2026-10-28 both tells could pass the check: about one round in 150 ran
  `AfterTrial` twice and failed the second tell with `UpdateFinishedTrialError`, even with
  `skipIfFinished`. Tells through two storage instances or processes are arbitrated by the storage, as
  in Python, and the loser gets `UpdateFinishedTrialError`.

Limits and costs:

- **Throughput.** The file journal syncs every append to disk (`fsync`), so many workers on a fast
  objective are bound by the disk. 16 contending workers sustained about 80–150 appends per second on
  an SSD in testing. Redis is about 5–8× faster at the same contention. On a standalone Redis server
  OptunaSharp reads the journal a thousand records per round trip and seeds the counter inside its append
  script, so a two-parameter trial costs 12 client round trips instead of Python's 20: 92–127 trials per
  second against 62–69 before (local Redis, 2026-11-07). A [gRPC proxy](#grpc-storage-proxy-distributed-cross-language)
  in front of the file gives one writer process.
- **Crashed workers.** The journal has no heartbeat, in Python too. A trial whose process dies stays
  `RUNNING`.
- **Repeated candidates.** Every Core sampler completes every trial under `nJobs`: all 32 were checked at
  `nJobs: 16` on a journal file (SAASBO, whose model is slow, at `nJobs: 8`). Draws from a sampler's random
  generator never repeat. Since 2026-11-08 no sampler hands one point to two trials asked at once either: the
  samplers that derived a candidate from state that changes only when a trial finishes (JANUS, JANUS-DE, PSO,
  DE, LM-CMA, TuRBO's and SAASBO's startup designs, REMBO, the genetic samplers, BruteForce) now also look at
  the trials still running, which changes nothing in a run of one trial at a time
  ([THREADING.md §6.3](https://github.com/SciSharp/OptunaSharp/blob/main/docs/THREADING.md)). Whole repeated
  points in 120 trials at `nJobs: 16` (in memory, three runs), against one trial at a time:
  - TPE, Random, CMA-ES, VkD-CMA, QMC, PartialFixed, OpenAI-ES, SNES, JANUS, JANUS-DE, GP, TuRBO and REMBO:
    none. JANUS used to repeat 30 without `nJobs` lanes, JANUS-DE 29, REMBO 51, LM-CMA 24 and TuRBO 4. xNES can
    degenerate on this kind of bounded problem, one trial at a time too, and then repeats points clamped onto
    the box's corners.
  - The genetic samplers (NSGA-II/III, SMS-EMOA, AGE-MOEA-2, RVEA, MOEA/D and the JANUS multi-objective
    family), DE, PSO and LM-CMA copy a finished point as often in parallel as one at a time, by design: a child
    that copies its parent, a mutant clamped onto a corner of the box (JANUS-DE's DE host too), a particle at
    rest. NSGA-II repeated 10–25 points against 16, LM-CMA 7–13 against 9, DE 0–3 against 2.
  - CMA-ES and VkD-CMA seed each trial's sampling with `randint(1, 65536) + trial.number`, as Optuna does, so
    two trials of one generation share a point about once in 65,535 pairs, one at a time too.
  - Grid can hand out a point twice once every grid point is running or finished, as Optuna documents, and
    still covers every point before it stops the study (a 90-point grid took 104–107 trials). BruteForce no
    longer hands out a point a running trial holds, unlike Optuna's `avoidPrematureStop` mode, so the trials
    asked once every point is taken re-evaluate finished points instead (91–101 trials).
  - QMC keys its sequence counter by the search space, which differs while the first trials are still
    running, so early concurrent trials reuse sequence positions, as Python's `QMCSampler` does. Their points
    still differ in the other coordinates.
  - Two processes on one storage are kept apart only partly: PSO's particles, DE's targets and TuRBO's and
    SAASBO's design rows are stamped in the storage, but JANUS's slots and the redraws of a point another trial
    holds are kept inside one sampler instance.
- **StackExchange.Redis and the thread pool.** StackExchange.Redis 3.x completes even synchronous calls
  on the .NET thread pool. If your code blocks more pool threads than the pool's minimum on journal
  calls at once, Redis calls time out after `SyncTimeout`. Examples: dozens of `Task.Run(() =>
  study.Optimize(...))`, an ASP.NET request burst, or `Optimize` with the opt-in thread-pool executor at an
  `nJobs` at or above the pool's minimum. The StackExchange.Redis and NRedisStack backends therefore turn
  on StackExchange.Redis's process-wide `DedicatedThreads` feature flag before they connect, unless you pass
  `dedicatedThreads: false` or your own multiplexer: 64 callers then told 640 of 640 trials in 7 s, against
  634 in 37 s with timeouts. The flag affects every StackExchange.Redis connection the process opens
  afterwards, and the .NET 8 builds (StackExchange.Redis 2.x) have no such flag; the
  [Redis guide](../integrations/storage/redis/index.md#thread-pool-callers-and-the-dedicatedthreads-flag)
  has the details. Otherwise use dedicated threads, as `nJobs` does by default, or raise
  `ThreadPool.SetMinThreads`.

### The same setups on SQLite, PostgreSQL, MySQL and gRPC

The storage suite's `StorageConcurrencyTests` runs four of these setups on the other shared storages. Each
must produce a perfect study:

1. one storage under `nJobs: 8`;
2. eight separate storages, each with its own connection, optimizing one study;
3. eight storages popping enqueued trials;
4. sixteen storages creating one study name on a fresh database.

How each storage gets there:

- **SQLite.** Each storage serializes its own connection, and separate storages wait on SQLite's database
  lock (the busy timeout is 30 s). An enqueued trial is popped by exactly one storage. Optuna 5.0 on SQLite
  can pop it twice: SQLite ignores the `SELECT … FOR UPDATE` its pop relies on. The second worker's tell
  then fails, and the dead workers leave queued trials `WAITING`.
- **PostgreSQL and MySQL (EF Core).** Trial creation holds the study's row lock, as Optuna's does, so
  concurrent creators number their trials one after another. Before 2026-10-28 they could take the same
  number (see the [EF Core page](../integrations/storage/ef-core.md#concurrency-heartbeats-and-schema-lifecycle)).
  On a fresh PostgreSQL database, schema creation is serialized, so a fleet opening it at once succeeds;
  Optuna fails that case 15 times in 16.
- **gRPC proxy.** Every client's call reaches the server's storage, which arbitrates between them;
  separate clients behave as separate processes on that storage.

## Caching over a backend

`CachedStorage` wraps another storage that implements `ICachedStorageBackend`, serving reads from an
in-process cache and writing through to the backend. It reduces round-trips to a slow or remote
backend during a long run.

```csharp
using OptunaSharp.Storages;

var cached = new CachedStorage(backend);   // backend : ICachedStorageBackend
```

## Resilience: heartbeats and failed-trial retry

A worker that dies mid-trial leaves that trial `Running` forever. Storages that support a
**heartbeat** can detect such stale trials and hand them to a callback. `RetryHeartbeatStaleTrialCallback`
re-enqueues a stale trial's parameters as a fresh trial:

```csharp
using OptunaSharp.Storages;

var callback = new RetryHeartbeatStaleTrialCallback(maxRetry: 3);
study.Optimize(Objective, nTrials: 100, callbacks: new[] { callback.Invoke });

// Inspecting a retried trial:
int? original = RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(trial);
```

`RetryFailedTrialCallback` is a deprecated alias of the same type.

## Resolving a storage from a value or URL

`StorageFunctions.GetStorage` is the helper `Study` uses internally and is handy in library code:

```csharp
using OptunaSharp.Storages;

BaseStorage s1 = StorageFunctions.GetStorage((BaseStorage?)null); // -> new InMemoryStorage()
BaseStorage s2 = StorageFunctions.GetStorage(existingStorage); // returned as-is
// RDB URL strings throw NotSupportedException; construct the satellite backend explicitly.
// redis:// URLs throw ValueError (construct JournalRedisBackend explicitly instead).
```

## Disposing a storage

Every storage that owns a connection implements both `IDisposable` and `IAsyncDisposable`: the SQLite and
EF Core RDB storages, the gRPC proxy, and the three Redis journal backends. Use `using` in synchronous code
and `await using` in async code. A DI container disposes the ones it builds with their scope or container,
and an asynchronously disposed scope (an ASP.NET Core request scope, `AsyncServiceScope`) uses
`DisposeAsync`. Only what a storage created is closed: an injected connection, data source, channel or
client stays the caller's. Disposing twice, by either method, does nothing. That matters with a scoped or
transient registration: the container disposes the instance once through its concrete registration and
once through the `BaseStorage` forwarder. `InMemoryStorage`, `JournalStorage` and the journal file backend
hold nothing to dispose.

`DisposeAsync` frees the calling thread where the client can close asynchronously: StackExchange.Redis,
NRedisStack, ServiceStack.Redis and Npgsql. Microsoft.Data.Sqlite and gRPC channels close synchronously
in either form.

## The `BaseStorage` contract

Every storage derives from the abstract `BaseStorage`. You rarely call it directly — `Study` does —
but the surface matters when writing a custom backend. It covers study lifecycle
(`CreateNewStudy`, `DeleteStudy`, `GetStudyIdFromName`, `GetStudyDirections`, `GetAllStudies`, …),
trial lifecycle (`CreateNewTrial`, `SetTrialParam`, `SetTrialStateValues`,
`SetTrialIntermediateValue`, `GetTrial`, `GetAllTrials`, …), and attribute setters for both. New study
names default to the `no-name-` prefix (`BaseStorage.DefaultStudyNamePrefix`), matching Optuna.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.storages.InMemoryStorage()` | `new InMemoryStorage()` |
| `JournalStorage(JournalFileBackend("study.log"))` | `new JournalStorage(new JournalFileBackend("study.log"))` |
| `JournalStorage(JournalRedisBackend("redis://…"))` | `new JournalStorage(new StackExchangeJournalRedisBackend("redis://…"))` |
| `optuna.storages.RDBStorage("sqlite:///optuna.db")` | `new RDBStorage("sqlite:///optuna.db")` |
| `optuna.storages.GrpcStorageProxy(host=…, port=…)` | `new GrpcStorageProxy(host: …, port: …)` |
| `optuna.storages.run_grpc_proxy_server(storage, …)` | `GrpcProxyServer.RunGrpcProxyServer(storage, …)` |
| `optuna.storages.RetryFailedTrialCallback` | `RetryHeartbeatStaleTrialCallback` |

**Divergence.** Journal *snapshots* are serialized in OptunaSharp's own binary format rather than Python
`pickle` bytes, so a snapshot cannot be resumed across languages (the *log* itself is fully
cross-language). This is a deliberate, documented difference — pickle compatibility is a non-goal. The
JSON format it replaced on 2026-11-07 still restores, but older OptunaSharp versions cannot read the new one. On
Redis each language keeps its snapshot under its own key, so neither overwrites the other's, and both save
one every 100 ids, Python's interval.

## See also

- [Study and Trials](study-and-trials.md) — how a study uses its storage.
- [Artifacts](artifacts.md) — large-file storage attached to trials, a separate concern from the study store.
- The generated [API reference](../api/index.md) for the full `OptunaSharp.Storages` surface.
