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
| `JournalStorage` + `JournalRedisBackend` | Yes (Redis) | Distributed workers | Yes (shared Redis keys) | `OptunaSharp.Storage.Redis` |
| `CachedStorage` | Wraps another | Read-through cache over a backend | Inherited | Core |

> RDB storage (SQLite / PostgreSQL / MySQL) and the gRPC storage proxy are on the roadmap
> (Phases 4.1–4.4) and are not yet available. `StorageFunctions.GetStorage` throws a clear error for
> `sqlite:///`, `postgresql://`, and similar URLs today.

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

- `JournalFileSymlinkLock` — portable, uses an atomic symlink (the default on systems without
  reliable `O_EXCL` open semantics).
- `JournalFileOpenLock` — uses exclusive file creation; preferred on POSIX file systems.

```csharp
var backend = new JournalFileBackend("study.log", new JournalFileOpenLock("study.log"));
```

Both accept an optional `gracePeriod` (seconds, default `30`) after which a stale lock left by a
crashed process is reclaimed.

## Redis journal storage (distributed)

For distributed workers sharing state through Redis, install the **`OptunaSharp.Storage.Redis`**
satellite package and use `JournalRedisBackend` (namespace `OptunaSharp.Storages.Journal`). It carries
the same journal semantics as the file backend, so no extra study code changes.

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalRedisBackend("redis://localhost:6379/0"));
var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
```

`JournalRedisBackend(string url, bool useCluster = false, string prefix = "")` accepts `redis://`,
`rediss://` (TLS), and `unix://` URLs. Set `useCluster: true` for a Redis Cluster; use `prefix` to
namespace multiple studies in one Redis instance. The keys and JSON payloads are shared with Python
Optuna's Redis journal backend.

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

BaseStorage s1 = StorageFunctions.GetStorage(null);            // -> new InMemoryStorage()
BaseStorage s2 = StorageFunctions.GetStorage(existingStorage); // returned as-is
// RDB URLs (sqlite:///..., postgresql://...) throw NotSupported until Phase 4.1;
// redis:// URLs throw ValueError (construct JournalRedisBackend explicitly instead).
```

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
| `JournalStorage(JournalRedisBackend("redis://…"))` | `new JournalStorage(new JournalRedisBackend("redis://…"))` |
| `optuna.storages.RetryFailedTrialCallback` | `RetryHeartbeatStaleTrialCallback` |

**Divergence.** Journal *snapshots* are serialized as JSON rather than Python `pickle` bytes, so a
snapshot cannot be resumed across languages (the *log* itself is fully cross-language). This is a
deliberate, documented difference — pickle compatibility is a non-goal.

## See also

- [Study and Trials](study-and-trials.md) — how a study uses its storage.
- [Artifacts](artifacts.md) — large-file storage attached to trials, a separate concern from the study store.
- The generated [API reference](../api/index.md) for the full `OptunaSharp.Storages` surface.
