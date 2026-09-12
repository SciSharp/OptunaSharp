# NRedisStack

`OptunaSharp.Storage.NRedisStack` provides `NRedisStackJournalRedisBackend` in `OptunaSharp.Storages.Journal`. It fits applications that already depend on NRedisStack and want to use their existing Redis connection for Optuna study persistence.

## Install and use

Requires .NET 8 or .NET 10 and a reachable Redis server. The journal itself requires only ordinary Redis string and Lua commands; Redis Stack modules are not needed for its storage operations.

```bash
dotnet add package OptunaSharp.Storage.NRedisStack
```

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

using var connection = ConnectionMultiplexer.Connect(new ConfigurationOptions
{
    EndPoints = { "localhost:6379" },
    DefaultDatabase = 1,
    ClientName = "optunasharp-worker",
});

var backend = new NRedisStackJournalRedisBackend(connection, prefix: "experiments");
var study = Optuna.CreateStudy(
    storage: new JournalStorage(backend),
    studyName: "quadratic", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

NRedisStack brings StackExchange.Redis transitively, which is why configuration and connection types come from that namespace. The injected connection can also serve your application's NRedisStack module APIs.

## Constructors and transport

| First arguments | Behavior |
|---|---|
| `string url` | Parse a Redis URL and create a multiplexer |
| `string url, Action<ConfigurationOptions> configure` | Modify the parsed native options before connecting |
| `ConfigurationOptions options` | Use a complete native configuration |
| `IConnectionMultiplexer connection` | Borrow an existing application connection |

Every overload accepts `useCluster = false` and `prefix = ""`. URL forms include `redis://`, `rediss://` for TLS, and `unix://`. URL database selection sets `DefaultDatabase`; journal operations use the multiplexer's default database. URL-derived options set `AbortOnConnectFail = false`, while explicitly supplied options retain your settings.

For example, the convenience URL callback can set timeouts:

```csharp
using OptunaSharp.Storages.Journal;

var backend = new NRedisStackJournalRedisBackend(
    "redis://localhost:6379/1",
    options =>
    {
        options.ConnectTimeout = 5000;
        options.SyncTimeout = 5000;
    },
    prefix: "experiments");
```

The URL and configuration constructors create a multiplexer internally, but the backend has no disposal hook. Use an application-owned multiplexer, as in the first example, for explicit lifetime management.

## Why the journal does not use RedisJSON

NRedisStack's module APIs do not change this backend's storage format. It uses the underlying `IDatabase` for `StringGet`, `StringSet`, `StringIncrement`, and `ScriptEvaluate`; snapshots are raw byte values. Logs are ordinary JSON strings encoded by OptunaSharp, not RedisJSON documents, search indexes, or time-series entries.

That choice keeps log bytes interchangeable with [StackExchange.Redis](stackexchange-redis.md), [ServiceStack.Redis](servicestack-redis.md), and Python Optuna. The [Redis overview](index.md) explains the remaining boundary: replay snapshots are runtime-specific, and cluster mode separates counter allocation from the log write. Construction emits an experimental warning.

Implementation: [NRedisStackJournalRedisBackend](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.NRedisStack/Storages/Journal/NRedisStackJournalRedisBackend.cs). Verification covers [native constructor forms](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/RedisBackendTests.cs), [cross-library byte parity](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/CrossLibraryParityTests.cs), and [Python journal interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs).
