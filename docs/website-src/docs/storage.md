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

- `JournalFileSymlinkLock` — the default; uses a symlink and may require extra privileges on Windows.
- `JournalFileOpenLock` — uses exclusive file creation; useful on Windows when symlink privileges
  are unavailable, provided the filesystem supports the required exclusive-create semantics.

```csharp
var backend = new JournalFileBackend("study.log", new JournalFileOpenLock("study.log"));
```

Both accept an optional `gracePeriod` (seconds, default `30`) after which a stale lock left by a
crashed process is reclaimed.

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
Journal logs can be shared with Python Optuna; snapshots use different serialization formats.
The [Redis guide](../integrations/storage/redis/index.md) explains those boundaries.

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

**Divergence.** Journal *snapshots* are serialized as JSON rather than Python `pickle` bytes, so a
snapshot cannot be resumed across languages (the *log* itself is fully cross-language). This is a
deliberate, documented difference — pickle compatibility is a non-goal.

## See also

- [Study and Trials](study-and-trials.md) — how a study uses its storage.
- [Artifacts](artifacts.md) — large-file storage attached to trials, a separate concern from the study store.
- The generated [API reference](../api/index.md) for the full `OptunaSharp.Storages` surface.
