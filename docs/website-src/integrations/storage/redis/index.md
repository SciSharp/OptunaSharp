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

`rediss://` enables TLS. The Unix-socket form is supported by StackExchange.Redis and NRedisStack; ServiceStack rejects it.

The shared parser reads a URL exactly as redis-py 8's `Redis.from_url` does:

- **Scheme:** it must be exactly `redis://`, `rediss://` or `unix://`.
- **User information:** split at its last `@`. It wins over `?username=`/`?password=`.
- **Host:** lower-cased with Python's `str.lower()` (a word-final `Σ` becomes `ς`, `İ` becomes `i̇`), then percent-decoded. `?host=` and `?port=` fill a host or port the URL leaves out.
- **Database:** `?db=`, else the path with its slashes removed.
- **Query values:** form-decoded (`+` is a space). The first of a repeated key wins. Typed keys use Python's `int`/`float` and redis-py's `to_bool`.

redis-py's connection options map onto every client:

- `socket_timeout` and `socket_connect_timeout`, which default to redis-py's 5 seconds
- `client_name`, `lib_name` and `protocol`
- the TLS settings `ssl_cert_reqs`, `ssl_check_hostname`, `ssl_ca_certs`, `ssl_ca_data`, `ssl_ca_path`, `ssl_certfile`, `ssl_keyfile`, `ssl_password` and `ssl_min_version`

TLS follows Python's `ssl` module. The system's certificate authorities are trusted plus the URL's. The host name is checked unless turned off. Revocation is never checked. An `ssl_keyfile` needs an `ssl_certfile`: without one, redis-py's `load_cert_chain` fails with `TypeError: certfile should be a valid filesystem path`, and so does the URL here. An argument redis-py's connection class does not take fails with redis-py's `TypeError` text, for example `AbstractConnection.__init__() got an unexpected keyword argument 'timeout'`. Other redis-py rejections keep redis-py's messages.

Options that only tune redis-py's own pool, retries or decoding are accepted without effect, for example `max_connections`, `retry_on_timeout` and `decode_responses`. Options no .NET client can honor raise `NotSupportedException`:

- arguments that take a Python object (`retry`, `credential_provider`, …)
- a text encoding other than UTF-8
- `ssl_ciphers`, the OpenSSL verification flags, and stapled OCSP
- a zero timeout

redis-py would only discover some values at the server, such as a client name with a space or a negative database index. Those are refused up front, with the server's message. ServiceStack.Redis also refuses what it cannot set per connection: RESP3, a user name without a password, certificate settings beyond the default check, and a minimum TLS version. Settings beyond the URL belong in the callback or the native client configuration.

All three backends implement `IDisposable` and `IAsyncDisposable`, and ownership follows how you constructed them:

- **Paths 1–3 (URL, URL plus callback, native configuration):** the backend opens the connection or pool, and `Dispose()` or `DisposeAsync()` closes it. Call one of them, or wrap the backend in `using` or `await using`, once every `JournalStorage` over it is done.
- **Path 4 (an injected client):** disposing the backend leaves the client open. It stays yours to dispose.

Disposal happens once: a second call to either method does nothing, and the two may even race on two threads. After it, the backend must not be used. A backend that closed its own connection or pool throws `ObjectDisposedException` on every later journal operation (ServiceStack's pool would otherwise keep serving after its own disposal, so that backend checks itself). A backend over an injected client keeps using it. `JournalStorage` itself has nothing to dispose.

`DisposeAsync()` closes without holding a thread. StackExchange.Redis and NRedisStack close the multiplexer through its own `DisposeAsync`, and ServiceStack closes its `RedisManagerPool` the same way. The synchronous `Dispose()` waits on the calling thread while the client shuts its sockets. In async code, prefer `await using`:

```csharp
await using var backend = new StackExchangeJournalRedisBackend("redis://localhost:6379", prefix: "experiments");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend), studyName: "s", loadIfExists: true);
```

The `Add…JournalStorage` DI methods take a `lifetime`. A backend the container builds is disposed with the scope that created it, or with the container for a singleton. A scoped or transient registration therefore closes its connection when its scope ends. (Before the backends were disposable, each scope leaked one open connection.) A scope disposed asynchronously, such as an ASP.NET Core request scope, an `AsyncServiceScope` or `await provider.DisposeAsync()`, calls `DisposeAsync()`. A scope disposed synchronously calls `Dispose()`. A pre-built backend that you register is never disposed by the container.

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
