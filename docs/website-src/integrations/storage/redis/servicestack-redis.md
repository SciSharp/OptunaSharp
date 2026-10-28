# ServiceStack.Redis

`OptunaSharp.Storage.ServiceStack.Redis` provides `ServiceStackJournalRedisBackend` in `OptunaSharp.Storages.Journal`. It uses a ServiceStack `IRedisClientsManager` and borrows clients for journal commands. This adapter connects an existing ServiceStack-based application to the same [journal format](index.md) as the other Redis adapters.

## Install and use

Requires .NET 8 or .NET 10 and a reachable Redis server. The package's [dependency notice](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.ServiceStack.Redis/README.md) identifies ServiceStack.Redis as commercially licensed with a request-limited free tier. Review the dependency's applicable terms for your application; OptunaSharp's own license does not replace them.

```bash
dotnet add package OptunaSharp.Storage.ServiceStack.Redis
```

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;
using ServiceStack.Redis;

using var manager = new RedisManagerPool("localhost:6379");
var backend = new ServiceStackJournalRedisBackend(manager, prefix: "experiments");
var study = Optuna.CreateStudy(
    storage: new JournalStorage(backend),
    studyName: "quadratic", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

The application owns the manager. The adapter returns borrowed `IRedisClient` instances after command use. Disposing the adapter does not dispose an injected manager. Keep the manager alive for the entire study workload.

## Constructors and endpoint settings

| First arguments | Behavior |
|---|---|
| `string url` | Parse the Redis URL and construct a `RedisManagerPool` |
| `string url, Action<ServiceStack.Redis.RedisEndpoint> configure` | Modify the native endpoint before pool creation |
| `ServiceStack.Redis.RedisEndpoint endpoint` | Construct a pool from the native endpoint |
| `IRedisClientsManager manager` | Borrow clients from your existing manager |

All overloads also accept `useCluster = false` and `prefix = ""`. `redis://` and `rediss://` URLs are supported. `unix://` throws `ValueError` because this adapter does not support Unix-domain sockets.

The native endpoint is distinct from `OptunaSharp.Storages.Journal.RedisEndpoint`, the neutral URL-parser result. Qualify or alias the ServiceStack type if both namespaces are imported:

```csharp
using OptunaSharp.Storages.Journal;
using ServiceStackEndpoint = ServiceStack.Redis.RedisEndpoint;

var backend = new ServiceStackJournalRedisBackend(
    new ServiceStackEndpoint("localhost", 6379) { Db = 1 },
    prefix: "experiments");
```

The endpoint and URL paths build a pool from the endpoint's `ToString()` connection representation. Host, port, TLS, database, username, and password flow through that representation. They do not expose a manager-configuration callback. Use an injected manager for pool selection, pool sizing, or application-managed connection behavior.

The implementation's documented ServiceStack caveat is that pooled-manager timeouts/retry use `RedisConfig` defaults rather than simply applying per-endpoint `ConnectTimeout`. Configure those settings where you manage ServiceStack. Leave the native endpoint's `NamespacePrefix` unset and use OptunaSharp's `prefix` argument to preserve the shared journal keys.

As with the other adapters, the backend is `IDisposable`. The URL and endpoint convenience constructors create an internal pool that the backend owns, and `Dispose()` closes it. An injected manager is never disposed by the backend.

## Commands, compatibility, and limits

The adapter uses `GetValue`, `SetValue`, `SetValueIfNotExists`, and `IncrementValue` for the raw journal strings. Standalone append uses `ExecLua` with the shared Lua script; cluster append allocates the counter and writes the record separately. Snapshots use native byte `Get`/`Set`, avoiding ServiceStack typed-object serialization.

`useCluster` selects that append algorithm; it does not provision a cluster or configure the manager's routing. Read the [Redis overview](index.md) for missing-record waits, prefix isolation, and the difference between interoperable logs and runtime-specific snapshots. Construction emits an experimental warning.

Implementation: [ServiceStackJournalRedisBackend](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Storage.ServiceStack.Redis/Storages/Journal/ServiceStackJournalRedisBackend.cs). Verification covers [pool/endpoint/callback constructors](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/RedisBackendTests.cs), [cross-library bytes](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Storage/CrossLibraryParityTests.cs), and [Python read/write interoperability](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs).
