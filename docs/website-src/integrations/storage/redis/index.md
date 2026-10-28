# Redis journal storage

The Redis integrations persist an Optuna operation journal in Redis. Each worker wraps a Redis backend in `JournalStorage`, which replays the journal and maintains study state in memory. Workers using the same Redis database and key prefix can see and extend the same studies.

## Packages and client choice

| Adapter package | Public backend type | Native client |
|---|---|---|
| [OptunaSharp.Storage.StackExchange.Redis](stackexchange-redis.md) | `StackExchangeJournalRedisBackend` | StackExchange.Redis |
| [OptunaSharp.Storage.NRedisStack](nredisstack.md) | `NRedisStackJournalRedisBackend` | NRedisStack, using StackExchange.Redis transport |
| [OptunaSharp.Storage.ServiceStack.Redis](servicestack-redis.md) | `ServiceStackJournalRedisBackend` | ServiceStack.Redis client manager |

Choose the adapter matching the Redis client your application manages. Each targets .NET 8 and .NET 10 and uses the same journal keys, JSON log encoding, append script, and binary snapshot storage. ServiceStack.Redis has separate dependency licensing terms, described on its adapter page.

## Start a study

Requires a reachable Redis server. Install one adapter:

```bash
dotnet add package OptunaSharp.Storage.StackExchange.Redis
```

This example makes the client lifetime explicit:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

using var connection = ConnectionMultiplexer.Connect("localhost:6379");
var backend = new StackExchangeJournalRedisBackend(
    connection, prefix: "optunasharp:experiments");
var storage = new JournalStorage(backend);

var study = Optuna.CreateStudy(
    storage: storage, studyName: "quadratic", loadIfExists: true);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);
```

Use the same database, prefix, and study name in the other workers. A prefix identifies a journal namespace, which can contain multiple named studies; it is not automatically derived from `studyName`.

## Connection configuration and lifetime

All three adapters expose four construction paths, each also accepting `useCluster = false` and `prefix = ""`:

1. A `redis-py`-style connection URL.
2. The URL plus a callback that changes the derived native configuration before connection creation.
3. Native configuration: `ConfigurationOptions` for StackExchange/NRedisStack, or `ServiceStack.Redis.RedisEndpoint` for ServiceStack.
4. An existing `IConnectionMultiplexer` or `IRedisClientsManager` owned by the application.

The URL parser supports these forms:

```text
redis://localhost:6379
redis://:password@localhost:6379/1
rediss://user:password@redis.example.com:6380/1
unix:///var/run/redis/redis.sock?db=1
```

`rediss://` enables TLS. The Unix-socket form is supported by StackExchange.Redis and NRedisStack; ServiceStack rejects it. The shared parser handles username, password, and database; supported query values (`db`, `username`, `password`) override URL components. Other native client settings belong in the callback or native client configuration, not arbitrary URL query parameters.

All three backends implement `IDisposable`, and ownership follows how you constructed them:

- **Paths 1–3 (URL, URL plus callback, native configuration):** the backend opens the connection or pool, and `Dispose()` closes it. Call it, or wrap the backend in `using`, once every `JournalStorage` over it is done.
- **Path 4 (an injected client):** `Dispose()` leaves the client open. It stays yours to dispose.

`Dispose()` can be called more than once; after it, the backend must not be used. `JournalStorage` itself has nothing to dispose.

The `Add…JournalStorage` DI methods take a `lifetime`. A backend the container builds is disposed with the scope that created it, or with the container for a singleton. A scoped or transient registration therefore closes its connection when its scope ends. (Before the backends were disposable, each scope leaked one open connection.) A pre-built backend that you register is never disposed by the container.

## Journal keys, atomicity, and recovery

| Key | Contents |
|---|---|
| `{prefix}:log_number` | Highest allocated log number; initialized to `-1` |
| `{prefix}:log:{n}` | One JSON operation record; first number is `0` |
| `{prefix}:snapshot` | Optional binary replay snapshot |

With the empty default prefix the keys begin with a colon, such as `:log_number`. Setting a prefix isolates a journal from other experiments using the same Redis database.

For standalone Redis, each log append uses one Lua `EVAL` that increments the counter and writes that record atomically. A list of logs is appended record by record; the whole list is not one transaction.

`useCluster: true` switches the append strategy to separate `INCR` and `SET` calls because the shared Lua script derives its keys internally. This flag does not discover nodes, configure client routing, or make the two writes atomic. Configure the native client's cluster connectivity separately. A connection failure between allocation and write can leave a missing record.

Readers first capture the highest log number, then read sequentially. A missing record is retried with exponential waits starting at 100 ms and capped at 10 seconds per wait, with no overall timeout. A malformed final JSON record stops that read; malformed earlier records raise an error. These behaviors make journal consistency and Redis data retention operationally important. The package does not configure Redis persistence, expiry, eviction, replication, or cluster recovery.

## Python interoperability and snapshot limits

The JSON operation logs are compatible with Python Optuna's `JournalRedisBackend`. Connect Python to the same Redis database and prefix:

```python
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalRedisBackend

storage = JournalStorage(JournalRedisBackend(
    "redis://localhost:6379", prefix="optunasharp:experiments"))
study = optuna.load_study(study_name="quadratic", storage=storage)
print(study.best_value)
```

Python needs Optuna and `redis`. The live interop tests write in Python, continue in C#, and read the result back in Python for all three adapters. Cross-library tests verify log bytes across all writer/reader pairs; the inherited backend contract tests separately verify raw snapshot bytes for each adapter. Cluster-branch tests use the shared Redis fixture, rather than exercising a multi-node cluster's routing or failover.

**Replay snapshots are not cross-language compatible.** OptunaSharp serializes JSON tagged `optunasharp-journal-snapshot-v1`; Python uses pickle. Both runtimes store snapshots under the same key. Their implementations can reject the other's snapshot and replay retained logs instead, so sharing the journal does not provide shared snapshot acceleration. Do not remove the operation history on the assumption that a snapshot is a portable replacement. The current live Redis language-interoperability test uses a short study; it does not itself exercise long histories crossing the snapshot interval.

## Shared package

`OptunaSharp.Storage.Redis` is a separate package installed transitively by the adapters. It has no Redis client dependency and cannot persist a study on its own. Its public `RedisUrl.Parse` produces a neutral `RedisEndpoint`; `RedisJournalProtocol` defines key builders, the initial counter, the append script, and the experimental-warning message. The adapters implement the journal operations independently using their native clients.

Construction of each adapter emits an `ExperimentalWarning`, following the upstream Redis journal API's experimental designation. The version in that warning refers to the upstream API introduction, not an OptunaSharp package version.

## Implementation and verification

- [Shared URL parser](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Redis/Storages/Journal/RedisEndpoint.cs) and [journal protocol](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.Redis/Storages/Journal/RedisJournalProtocol.cs)
- [Journal replay and snapshot implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Storages/Journal/JournalStorage.cs)
- [Adapter configuration tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/RedisBackendTests.cs) and [URL tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/RedisUrlTests.cs)
- [Backend contract, snapshot, and cluster-branch tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/JournalBackendContractTests.cs)
- [Cross-library byte parity](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/CrossLibraryParityTests.cs) and [Python journal interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs)
