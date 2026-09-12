# StackExchange.Redis

`OptunaSharp.Storage.StackExchange.Redis` provides `StackExchangeJournalRedisBackend` in `OptunaSharp.Storages.Journal`. It uses StackExchange.Redis `IDatabase` string commands and Lua execution to implement the shared [Redis journal format](index.md).

## Install and use

Requires .NET 8 or .NET 10 and a reachable Redis server.

```bash
dotnet add package OptunaSharp.Storage.StackExchange.Redis
```

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

using var connection = ConnectionMultiplexer.Connect(new ConfigurationOptions
{
    EndPoints = { "localhost:6379" },
    ConnectTimeout = 5000,
    SyncTimeout = 5000,
    ClientName = "optunasharp-worker",
});

var backend = new StackExchangeJournalRedisBackend(connection, prefix: "experiments");
var study = Optuna.CreateStudy(
    storage: new JournalStorage(backend),
    studyName: "quadratic", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

The multiplexer is application-owned. Keep it alive for all workers that use it and dispose it after their work finishes; the journal backend has no disposal method.

## Constructors and database selection

| First arguments | Behavior |
|---|---|
| `string url` | Parse `redis://`, `rediss://`, or `unix://`, then create a multiplexer |
| `string url, Action<ConfigurationOptions> configure` | Parse URL, apply callback, then connect |
| `ConfigurationOptions options` | Connect using the supplied native options |
| `IConnectionMultiplexer connection` | Borrow an existing connection |

All overloads also take `bool useCluster = false, string prefix = ""`. URL database selection sets `ConfigurationOptions.DefaultDatabase`; an injected connection uses its configured default database because the backend calls `GetDatabase()` without a database argument.

The URL path sets `AbortOnConnectFail = false` before invoking any callback. This permits constructing the connection while the server is unavailable, but the subsequent journal load or write still needs a working Redis connection. The native-options and injected-client paths preserve the configuration you supplied.

A URL callback can configure client timeouts or naming:

```csharp
using OptunaSharp.Storages.Journal;

var backend = new StackExchangeJournalRedisBackend(
    "redis://localhost:6379/1",
    options =>
    {
        options.ConnectTimeout = 5000;
        options.SyncTimeout = 5000;
        options.ClientName = "optunasharp-worker";
    },
    prefix: "experiments");
```

This convenience form creates an internal multiplexer. Use the injected-client form from the first example for deterministic shutdown. Native options expose TLS configuration, endpoint lists, and the other StackExchange.Redis connection settings; they do not alter the journal data model.

## Behavior and limits

`ReadLogs` uses `StringGet`; append initializes the counter with `StringSet(..., When.NotExists)` and uses `ScriptEvaluate` on standalone Redis. Cluster append uses `StringIncrement` followed by `StringSet`. Snapshots use raw Redis byte values.

Read the [Redis overview](index.md) for prefix isolation, non-atomic cluster writes, missing-record waits, and runtime-specific snapshots. NRedisStack and ServiceStack.Redis can access the same logs when pointed at the same database and prefix. Construction emits an experimental warning.

Implementation: [StackExchangeJournalRedisBackend](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.StackExchange.Redis/Storages/Journal/StackExchangeJournalRedisBackend.cs). Verification covers [all constructor forms](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/RedisBackendTests.cs), [cross-library bytes](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/CrossLibraryParityTests.cs), and [Python read/write interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs).
